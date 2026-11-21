#![forbid(unsafe_code)]

//! Safe IR block cloning for SPIR-V-derived Naga functions.
//!
//! Naga forbids sharing [`Statement::Emit`] expression ranges across multiple
//! execution paths because each `Handle<Expression>` must have a single defining
//! site dominating all uses. When lowering `loop { ... continuing { ... } }`
//! requires duplicating a `continuing` block across multiple `Continue` or
//! fallthrough sites, [`clone_block_with_fresh_emits`] clones every pure
//! `Emit`-defined expression into fresh slots in [`Arena<Expression>`] and
//! remaps all intra-block references consistently.

use std::collections::HashMap;

use naga::{
    Arena, AtomicFunction, Block, Expression, Handle, ImageQuery, Range, SampleLevel, Span,
    Statement, SwitchCase,
};

/// Returns `true` if `block` (or any nested structured block) contains
/// expression-defining statements or synchronization primitives that cannot be
/// safely duplicated by [`clone_block_with_fresh_emits`].
pub(super) fn block_contains_unclonable_defs(block: &Block) -> bool {
    for stmt in block.iter() {
        match stmt {
            Statement::WorkGroupUniformLoad { .. }
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::ImageAtomic { .. }
            | Statement::RayQuery { .. }
            | Statement::RayPipelineFunction(_)
            | Statement::CooperativeStore { .. } => return true,
            Statement::Call {
                result: Some(_), ..
            }
            | Statement::Atomic {
                result: Some(_), ..
            } => return true,
            Statement::Block(inner) => {
                if block_contains_unclonable_defs(inner) {
                    return true;
                }
            }
            Statement::If { accept, reject, .. } => {
                if block_contains_unclonable_defs(accept) || block_contains_unclonable_defs(reject)
                {
                    return true;
                }
            }
            Statement::Switch { cases, .. } => {
                if cases
                    .iter()
                    .any(|case| block_contains_unclonable_defs(&case.body))
                {
                    return true;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                if block_contains_unclonable_defs(body)
                    || block_contains_unclonable_defs(continuing)
                {
                    return true;
                }
            }
            Statement::Emit(_)
            | Statement::Call { result: None, .. }
            | Statement::Atomic { result: None, .. }
            | Statement::Break
            | Statement::Continue
            | Statement::Return { .. }
            | Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_)
            | Statement::Store { .. }
            | Statement::ImageStore { .. } => {}
        }
    }
    false
}

/// Clones `src_block` (and optionally remaps an associated condition handle such
/// as `Loop::break_if`) using freshly allocated `Handle<Expression>` entries in
/// `expressions` for every `Statement::Emit` range encountered inside
/// `src_block`.
///
/// # Errors
/// Returns an error if `src_block` contains any unclonable statement reported by
/// [`block_contains_unclonable_defs`].
pub(super) fn clone_block_with_fresh_emits(
    src_block: &Block,
    extra_expr: Option<Handle<Expression>>,
    expressions: &mut Arena<Expression>,
) -> Result<(Block, Option<Handle<Expression>>), String> {
    if block_contains_unclonable_defs(src_block) {
        return Err(String::from(
            "cannot clone block containing non-Emit expression definitions across switch continues",
        ));
    }

    let mut remap = HashMap::new();
    let cloned_block = clone_block_inner(src_block, expressions, &mut remap)?;
    let cloned_extra = extra_expr.map(|h| remap_handle(&remap, h));
    Ok((cloned_block, cloned_extra))
}

fn clone_block_inner(
    src_block: &Block,
    expressions: &mut Arena<Expression>,
    remap: &mut HashMap<Handle<Expression>, Handle<Expression>>,
) -> Result<Block, String> {
    let mut dst = Block::new();
    for (stmt, &span) in src_block.span_iter() {
        let cloned_stmt = match stmt {
            Statement::Emit(range) => {
                if let Some(new_range) = clone_emit_range(range.clone(), span, expressions, remap) {
                    Statement::Emit(new_range)
                } else {
                    continue;
                }
            }
            Statement::Block(inner) => {
                Statement::Block(clone_block_inner(inner, expressions, remap)?)
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => Statement::If {
                condition: remap_handle(remap, *condition),
                accept: clone_block_inner(accept, expressions, remap)?,
                reject: clone_block_inner(reject, expressions, remap)?,
            },
            Statement::Switch { selector, cases } => {
                let mut cloned_cases = Vec::with_capacity(cases.len());
                for case in cases {
                    cloned_cases.push(SwitchCase {
                        value: case.value,
                        body: clone_block_inner(&case.body, expressions, remap)?,
                        fall_through: case.fall_through,
                    });
                }
                Statement::Switch {
                    selector: remap_handle(remap, *selector),
                    cases: cloned_cases,
                }
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => Statement::Loop {
                body: clone_block_inner(body, expressions, remap)?,
                continuing: clone_block_inner(continuing, expressions, remap)?,
                break_if: break_if.map(|h| remap_handle(remap, h)),
            },
            Statement::Break => Statement::Break,
            Statement::Continue => Statement::Continue,
            Statement::Kill => Statement::Kill,
            Statement::ControlBarrier(barrier) => Statement::ControlBarrier(*barrier),
            Statement::MemoryBarrier(barrier) => Statement::MemoryBarrier(*barrier),
            Statement::Return { value } => Statement::Return {
                value: value.map(|h| remap_handle(remap, h)),
            },
            Statement::Store { pointer, value } => Statement::Store {
                pointer: remap_handle(remap, *pointer),
                value: remap_handle(remap, *value),
            },
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => Statement::ImageStore {
                image: remap_handle(remap, *image),
                coordinate: remap_handle(remap, *coordinate),
                array_index: array_index.map(|h| remap_handle(remap, h)),
                value: remap_handle(remap, *value),
            },
            Statement::Call {
                function,
                arguments,
                result: None,
            } => Statement::Call {
                function: *function,
                arguments: arguments.iter().map(|h| remap_handle(remap, *h)).collect(),
                result: None,
            },
            Statement::Atomic {
                pointer,
                fun,
                value,
                result: None,
            } => Statement::Atomic {
                pointer: remap_handle(remap, *pointer),
                fun: remap_atomic_fun(remap, *fun),
                value: remap_handle(remap, *value),
                result: None,
            },
            Statement::Call {
                result: Some(_), ..
            }
            | Statement::Atomic {
                result: Some(_), ..
            }
            | Statement::WorkGroupUniformLoad { .. }
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::ImageAtomic { .. }
            | Statement::RayQuery { .. }
            | Statement::RayPipelineFunction(_)
            | Statement::CooperativeStore { .. } => {
                return Err(String::from(
                    "encountered unclonable statement while duplicating loop block",
                ));
            }
        };
        dst.push(cloned_stmt, span);
    }
    Ok(dst)
}

fn clone_emit_range(
    range: Range<Expression>,
    span: Span,
    expressions: &mut Arena<Expression>,
    remap: &mut HashMap<Handle<Expression>, Handle<Expression>>,
) -> Option<Range<Expression>> {
    let mut first_new = None;
    let mut last_new = None;
    for old_handle in range {
        let mut expr = expressions[old_handle].clone();
        remap_expression(&mut expr, remap);
        let new_handle = expressions.append(expr, span);
        remap.insert(old_handle, new_handle);
        if first_new.is_none() {
            first_new = Some(new_handle);
        }
        last_new = Some(new_handle);
    }
    match (first_new, last_new) {
        (Some(first), Some(last)) => Some(Range::new_from_bounds(first, last)),
        _ => None,
    }
}

fn remap_handle(
    remap: &HashMap<Handle<Expression>, Handle<Expression>>,
    handle: Handle<Expression>,
) -> Handle<Expression> {
    remap.get(&handle).copied().unwrap_or(handle)
}

fn remap_opt(
    remap: &HashMap<Handle<Expression>, Handle<Expression>>,
    opt: &mut Option<Handle<Expression>>,
) {
    if let Some(handle) = opt {
        *handle = remap_handle(remap, *handle);
    }
}

fn remap_atomic_fun(
    remap: &HashMap<Handle<Expression>, Handle<Expression>>,
    fun: AtomicFunction,
) -> AtomicFunction {
    match fun {
        AtomicFunction::Exchange { compare } => AtomicFunction::Exchange {
            compare: compare.map(|h| remap_handle(remap, h)),
        },
        other => other,
    }
}

fn remap_expression(
    expr: &mut Expression,
    remap: &HashMap<Handle<Expression>, Handle<Expression>>,
) {
    match expr {
        Expression::Compose { components, .. } => {
            for component in components {
                *component = remap_handle(remap, *component);
            }
        }
        Expression::Access { base, index } => {
            *base = remap_handle(remap, *base);
            *index = remap_handle(remap, *index);
        }
        Expression::AccessIndex { base, .. } => {
            *base = remap_handle(remap, *base);
        }
        Expression::Splat { value, .. } => {
            *value = remap_handle(remap, *value);
        }
        Expression::Swizzle { vector, .. } => {
            *vector = remap_handle(remap, *vector);
        }
        Expression::Load { pointer } => {
            *pointer = remap_handle(remap, *pointer);
        }
        Expression::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            ..
        } => {
            *image = remap_handle(remap, *image);
            *sampler = remap_handle(remap, *sampler);
            *coordinate = remap_handle(remap, *coordinate);
            remap_opt(remap, array_index);
            remap_opt(remap, offset);
            match level {
                SampleLevel::Exact(h) | SampleLevel::Bias(h) => {
                    *h = remap_handle(remap, *h);
                }
                SampleLevel::Gradient { x, y } => {
                    *x = remap_handle(remap, *x);
                    *y = remap_handle(remap, *y);
                }
                SampleLevel::Auto | SampleLevel::Zero => {}
            }
            remap_opt(remap, depth_ref);
        }
        Expression::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            *image = remap_handle(remap, *image);
            *coordinate = remap_handle(remap, *coordinate);
            remap_opt(remap, array_index);
            remap_opt(remap, sample);
            remap_opt(remap, level);
        }
        Expression::ImageQuery { image, query } => {
            *image = remap_handle(remap, *image);
            if let ImageQuery::Size { level } = query {
                remap_opt(remap, level);
            }
        }
        Expression::Unary { expr: inner, .. } => {
            *inner = remap_handle(remap, *inner);
        }
        Expression::Binary { left, right, .. } => {
            *left = remap_handle(remap, *left);
            *right = remap_handle(remap, *right);
        }
        Expression::Select {
            condition,
            accept,
            reject,
        } => {
            *condition = remap_handle(remap, *condition);
            *accept = remap_handle(remap, *accept);
            *reject = remap_handle(remap, *reject);
        }
        Expression::Derivative { expr: inner, .. } => {
            *inner = remap_handle(remap, *inner);
        }
        Expression::Relational { argument, .. } => {
            *argument = remap_handle(remap, *argument);
        }
        Expression::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            *arg = remap_handle(remap, *arg);
            remap_opt(remap, arg1);
            remap_opt(remap, arg2);
            remap_opt(remap, arg3);
        }
        Expression::As { expr: inner, .. } => {
            *inner = remap_handle(remap, *inner);
        }
        Expression::ArrayLength(inner) => {
            *inner = remap_handle(remap, *inner);
        }
        Expression::RayQueryVertexPositions { query, .. }
        | Expression::RayQueryGetIntersection { query, .. } => {
            *query = remap_handle(remap, *query);
        }
        Expression::CooperativeMultiplyAdd { a, b, c } => {
            *a = remap_handle(remap, *a);
            *b = remap_handle(remap, *b);
            *c = remap_handle(remap, *c);
        }
        Expression::Literal(_)
        | Expression::Constant(_)
        | Expression::Override(_)
        | Expression::ZeroValue(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. }
        | Expression::WorkGroupUniformLoadResult { .. }
        | Expression::RayQueryProceedResult
        | Expression::SubgroupBallotResult
        | Expression::SubgroupOperationResult { .. }
        | Expression::CooperativeLoad { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use naga::{Arena, BinaryOperator, Block, Expression, Literal, Range, Span, Statement};

    use super::{block_contains_unclonable_defs, clone_block_with_fresh_emits};

    #[test]
    fn clones_emit_chain_with_distinct_handles_and_rejects_unclonable_ops() {
        let mut exprs = Arena::new();
        let lit_one = exprs.append(Expression::Literal(Literal::U32(1)), Span::default());
        let sum = exprs.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: lit_one,
                right: lit_one,
            },
            Span::default(),
        );
        let mut block = Block::new();
        block.push(
            Statement::Emit(Range::new_from_bounds(sum, sum)),
            Span::default(),
        );

        let (cloned, cloned_sum) =
            clone_block_with_fresh_emits(&block, Some(sum), &mut exprs).expect("clone succeeds");
        let new_sum = cloned_sum.expect("extra handle remapped");
        assert_ne!(new_sum, sum);
        assert_eq!(cloned.len(), 1);
        assert!(matches!(
            exprs[new_sum],
            Expression::Binary {
                op: BinaryOperator::Add,
                left,
                right
            } if left == lit_one && right == lit_one
        ));

        let mut unclonable = Block::new();
        unclonable.push(
            Statement::WorkGroupUniformLoad {
                pointer: lit_one,
                result: lit_one,
            },
            Span::default(),
        );
        assert!(block_contains_unclonable_defs(&unclonable));
        assert!(clone_block_with_fresh_emits(&unclonable, None, &mut exprs).is_err());
    }
}
