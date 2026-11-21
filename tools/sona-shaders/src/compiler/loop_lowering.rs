#![forbid(unsafe_code)]

//! Naga IR loop lowering pass removing all non-empty `continuing` blocks and `break if` guards.
//!
//! `naga 30.0.2`'s Metal Shading Language writer emits `continuing` and `break_if` at the top of
//! `while(true)` gated by `if (!loop_init)` before evaluating `body`. Combined with SPIR-V->WGSL
//! `Load` re-inlining (`bake_ref_count() == 1`), `break_if` re-reads mutated loop-carrier locals
//! instead of loop-entry values. Flattening `continuing` and `break_if` inline into `body`
//! eliminates `loop_init` across all compiled kernels (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod clone_ir;

use anyhow::{Context as _, bail};
use naga::{
    Arena, Block, Expression, Handle, Literal, LocalVariable, Module, Range, Scalar, Span,
    Statement, Type, TypeInner, valid,
};

fn block_always_diverges_from_loop(block: &Block, in_switch: bool) -> bool {
    block
        .iter()
        .any(|stmt| stmt_always_diverges_from_loop(stmt, in_switch))
}

fn stmt_always_diverges_from_loop(stmt: &Statement, in_switch: bool) -> bool {
    match stmt {
        Statement::Break => !in_switch,
        Statement::Continue | Statement::Return { .. } | Statement::Kill => true,
        Statement::Block(inner) => block_always_diverges_from_loop(inner, in_switch),
        Statement::If { accept, reject, .. } => {
            block_always_diverges_from_loop(accept, in_switch)
                && block_always_diverges_from_loop(reject, in_switch)
        }
        Statement::Switch { cases, .. } => {
            !cases.is_empty()
                && cases
                    .iter()
                    .filter(|case| !case.fall_through)
                    .all(|case| block_always_diverges_from_loop(&case.body, true))
        }
        Statement::Loop { .. } => false,
        _ => false,
    }
}

fn has_continue_in_switch(block: &Block, in_switch: bool) -> bool {
    for stmt in block.iter() {
        let hit = match stmt {
            Statement::Continue => in_switch,
            Statement::Block(inner) => has_continue_in_switch(inner, in_switch),
            Statement::If { accept, reject, .. } => {
                has_continue_in_switch(accept, in_switch)
                    || has_continue_in_switch(reject, in_switch)
            }
            Statement::Switch { cases, .. } => {
                cases.iter().any(|c| has_continue_in_switch(&c.body, true))
            }
            Statement::Loop { .. } => false,
            _ => false,
        };
        if hit {
            return true;
        }
    }
    false
}

fn count_depth0_continues(block: &Block) -> usize {
    let mut count = 0;
    for stmt in block.iter() {
        match stmt {
            Statement::Continue => count += 1,
            Statement::Block(inner) => count += count_depth0_continues(inner),
            Statement::If { accept, reject, .. } => {
                count += count_depth0_continues(accept);
                count += count_depth0_continues(reject);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    count += count_depth0_continues(&case.body);
                }
            }
            Statement::Loop { .. } => {}
            _ => {}
        }
    }
    count
}

fn move_tail_into_single_continue(block: &mut Block, tail: &mut Option<Block>) -> bool {
    let mut idx = 0;
    while idx < block.len() {
        match &mut block[idx] {
            Statement::Continue => {
                if let Some(mut replacement) = tail.take() {
                    replacement.push(Statement::Continue, Span::default());
                    block.splice(idx..=idx, replacement);
                    return true;
                }
                return false;
            }
            Statement::Block(inner) => {
                if move_tail_into_single_continue(inner, tail) {
                    return true;
                }
            }
            Statement::If { accept, reject, .. } => {
                if move_tail_into_single_continue(accept, tail)
                    || move_tail_into_single_continue(reject, tail)
                {
                    return true;
                }
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    if move_tail_into_single_continue(&mut case.body, tail) {
                        return true;
                    }
                }
            }
            Statement::Loop { .. } => {}
            _ => {}
        }
        idx += 1;
    }
    false
}

#[derive(Clone, Copy)]
enum ContinueTailAction {
    DirectBreakIf(Option<Handle<Expression>>),
    SwitchExitStore {
        ptr: Handle<Expression>,
        cond: Handle<Expression>,
    },
}

fn clone_tail_at_continues(
    block: &mut Block,
    source_continuing: &Block,
    action: ContinueTailAction,
    omit_trailing_continue: bool,
    exprs: &mut Arena<Expression>,
) -> anyhow::Result<()> {
    let extra_expr = match action {
        ContinueTailAction::DirectBreakIf(cond) => cond,
        ContinueTailAction::SwitchExitStore { cond, .. } => Some(cond),
    };
    let mut idx = 0;
    while idx < block.len() {
        let is_outer_last = omit_trailing_continue && idx + 1 == block.len();
        match &mut block[idx] {
            Statement::Continue => {
                let (mut replacement, remapped_extra) =
                    clone_ir::clone_block_with_fresh_emits(source_continuing, extra_expr, exprs)
                        .map_err(anyhow::Error::msg)?;
                match action {
                    ContinueTailAction::DirectBreakIf(_) => {
                        if let Some(cond_h) = remapped_extra {
                            replacement.push(
                                Statement::If {
                                    condition: cond_h,
                                    accept: Block::from_vec(vec![Statement::Break]),
                                    reject: Block::new(),
                                },
                                Span::default(),
                            );
                        }
                    }
                    ContinueTailAction::SwitchExitStore { ptr, .. } => {
                        let cond_h = remapped_extra
                            .expect("switch exit guard requires remapped break_if handle");
                        replacement.push(
                            Statement::Store {
                                pointer: ptr,
                                value: cond_h,
                            },
                            Span::default(),
                        );
                    }
                }
                if !is_outer_last {
                    replacement.push(Statement::Continue, Span::default());
                }
                let inserted = replacement.len();
                block.splice(idx..=idx, replacement);
                idx += inserted;
                continue;
            }
            Statement::Block(inner) => {
                clone_tail_at_continues(inner, source_continuing, action, false, exprs)?;
            }
            Statement::If { accept, reject, .. } => {
                clone_tail_at_continues(accept, source_continuing, action, false, exprs)?;
                clone_tail_at_continues(reject, source_continuing, action, false, exprs)?;
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    clone_tail_at_continues(
                        &mut case.body,
                        source_continuing,
                        action,
                        false,
                        exprs,
                    )?;
                }
            }
            Statement::Loop { .. } => {}
            _ => {}
        }
        idx += 1;
    }
    Ok(())
}

struct LoopLowerCtx<'a> {
    types: &'a mut naga::UniqueArena<Type>,
    locals: &'a mut Arena<LocalVariable>,
    exprs: &'a mut Arena<Expression>,
    next_local_id: &'a mut usize,
    entry_point: &'a str,
}

fn prepend_switch_exit_guard(body: &mut Block, load_expr: Handle<Expression>) {
    let mut prefix = Block::new();
    prefix.push(
        Statement::Emit(Range::new_from_bounds(load_expr, load_expr)),
        Span::default(),
    );
    prefix.push(
        Statement::If {
            condition: load_expr,
            accept: Block::from_vec(vec![Statement::Break]),
            reject: Block::new(),
        },
        Span::default(),
    );
    prefix.append(body);
    *body = prefix;
}

fn lower_single_loop(
    body: &mut Block,
    continuing: &mut Block,
    break_if: &mut Option<Handle<Expression>>,
    ctx: &mut LoopLowerCtx<'_>,
) -> anyhow::Result<Option<Statement>> {
    if continuing.is_empty() && break_if.is_none() {
        return Ok(None);
    }

    let continue_count = count_depth0_continues(body);
    let falls_through = !block_always_diverges_from_loop(body, false);
    let ends_with_continue = matches!(body.last(), Some(Statement::Continue));

    if continue_count == 0 && !falls_through {
        let _ = break_if.take();
        continuing.cull(0..);
        return Ok(None);
    }

    let has_switch_continue = break_if.is_some() && has_continue_in_switch(body, false);
    if has_switch_continue {
        let cond = break_if.take().expect("checked break_if.is_some()");
        let bool_ty = ctx.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::BOOL),
            },
            Span::default(),
        );
        let id = *ctx.next_local_id;
        *ctx.next_local_id += 1;
        let exit_local = ctx.locals.append(
            LocalVariable {
                name: Some(format!("_loop_exit_{id}")),
                ty: bool_ty,
                init: None,
            },
            Span::default(),
        );
        let ptr_expr = ctx
            .exprs
            .append(Expression::LocalVariable(exit_local), Span::default());
        let false_expr = ctx
            .exprs
            .append(Expression::Literal(Literal::Bool(false)), Span::default());
        let load_expr = ctx
            .exprs
            .append(Expression::Load { pointer: ptr_expr }, Span::default());

        let mut switch_tail = Block::new();
        if continue_count == 1 && !falls_through {
            switch_tail.append(continuing);
            switch_tail.push(
                Statement::Store {
                    pointer: ptr_expr,
                    value: cond,
                },
                Span::default(),
            );
            let mut boxed = Some(switch_tail);
            if !move_tail_into_single_continue(body, &mut boxed) {
                bail!(
                    "failed to locate single switch Continue target in '{}' loop",
                    ctx.entry_point
                );
            }
        } else {
            if clone_ir::block_contains_unclonable_defs(continuing) {
                bail!(
                    "cannot clone non-Emit expression-defining continuing block across switch continues \
                     in '{}' (continue_count={continue_count}, falls_through={falls_through})",
                    ctx.entry_point
                );
            }
            clone_tail_at_continues(
                body,
                continuing,
                ContinueTailAction::SwitchExitStore {
                    ptr: ptr_expr,
                    cond,
                },
                true,
                ctx.exprs,
            )?;
            if falls_through {
                body.append(continuing);
                body.push(
                    Statement::If {
                        condition: cond,
                        accept: Block::from_vec(vec![Statement::Break]),
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            } else {
                continuing.cull(0..);
            }
        }

        prepend_switch_exit_guard(body, load_expr);
        return Ok(Some(Statement::Store {
            pointer: ptr_expr,
            value: false_expr,
        }));
    }

    let raw_break_if = break_if.take();
    if continue_count == 1 && ends_with_continue && !falls_through {
        body.cull(body.len() - 1..);
        body.append(continuing);
        if let Some(cond) = raw_break_if {
            body.push(
                Statement::If {
                    condition: cond,
                    accept: Block::from_vec(vec![Statement::Break]),
                    reject: Block::new(),
                },
                Span::default(),
            );
        }
        return Ok(None);
    }
    if continue_count == 0 && falls_through {
        body.append(continuing);
        if let Some(cond) = raw_break_if {
            body.push(
                Statement::If {
                    condition: cond,
                    accept: Block::from_vec(vec![Statement::Break]),
                    reject: Block::new(),
                },
                Span::default(),
            );
        }
        return Ok(None);
    }
    if continue_count == 1 && !falls_through && !ends_with_continue {
        let mut boxed_tail = Block::new();
        boxed_tail.append(continuing);
        if let Some(cond) = raw_break_if {
            boxed_tail.push(
                Statement::If {
                    condition: cond,
                    accept: Block::from_vec(vec![Statement::Break]),
                    reject: Block::new(),
                },
                Span::default(),
            );
        }
        let mut opt_tail = Some(boxed_tail);
        if !move_tail_into_single_continue(body, &mut opt_tail) {
            bail!(
                "failed to locate single Continue target in '{}' loop body",
                ctx.entry_point
            );
        }
        return Ok(None);
    }
    if !clone_ir::block_contains_unclonable_defs(continuing) {
        clone_tail_at_continues(
            body,
            continuing,
            ContinueTailAction::DirectBreakIf(raw_break_if),
            true,
            ctx.exprs,
        )?;
        if falls_through {
            body.append(continuing);
            if let Some(cond) = raw_break_if {
                body.push(
                    Statement::If {
                        condition: cond,
                        accept: Block::from_vec(vec![Statement::Break]),
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        } else {
            continuing.cull(0..);
        }
        return Ok(None);
    }

    bail!(
        "cannot clone non-Emit expression definitions in loop continuing block for '{}' \
         (continue_count={continue_count}, falls_through={falls_through})",
        ctx.entry_point
    );
}

fn lower_block(block: &mut Block, ctx: &mut LoopLowerCtx<'_>) -> anyhow::Result<()> {
    let mut idx = 0;
    while idx < block.len() {
        let mut pre_stmt = None;
        match &mut block[idx] {
            Statement::Block(inner) => lower_block(inner, ctx)?,
            Statement::If { accept, reject, .. } => {
                lower_block(accept, ctx)?;
                lower_block(reject, ctx)?;
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    lower_block(&mut case.body, ctx)?;
                }
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                lower_block(body, ctx)?;
                lower_block(continuing, ctx)?;
                pre_stmt = lower_single_loop(body, continuing, break_if, ctx)?;
            }
            _ => {}
        }
        if let Some(init_store) = pre_stmt {
            block.splice(idx..idx, Block::from_vec(vec![init_store]));
            idx += 1;
        }
        idx += 1;
    }
    Ok(())
}

/// Lowers every Naga `Statement::Loop` `continuing` block and `break if` guard inline inside
/// `body` so Metal/WGSL downstream backends never emit broken `if (!loop_init)` pre-headers.
pub(crate) fn lower_wgsl_loop_continuing(
    wgsl: &str,
    entry_point: &str,
    uses_subgroups: bool,
) -> anyhow::Result<String> {
    let mut module: Module = naga::front::wgsl::parse_str(wgsl).map_err(|err| {
        anyhow::anyhow!(
            "WGSL parse failure prior to loop lowering for '{entry_point}':\n{}",
            err.emit_to_string(wgsl)
        )
    })?;

    let mut next_local_id = 0_usize;
    let types = &mut module.types;
    for (_, func) in module.functions.iter_mut() {
        let mut ctx = LoopLowerCtx {
            types,
            locals: &mut func.local_variables,
            exprs: &mut func.expressions,
            next_local_id: &mut next_local_id,
            entry_point,
        };
        lower_block(&mut func.body, &mut ctx)?;
    }
    for ep in &mut module.entry_points {
        let mut ctx = LoopLowerCtx {
            types,
            locals: &mut ep.function.local_variables,
            exprs: &mut ep.function.expressions,
            next_local_id: &mut next_local_id,
            entry_point,
        };
        lower_block(&mut ep.function.body, &mut ctx)?;
    }

    let caps = if uses_subgroups {
        valid::Capabilities::SUBGROUP | valid::Capabilities::SUBGROUP_BARRIER
    } else {
        valid::Capabilities::empty()
    };
    let mut validator = valid::Validator::new(valid::ValidationFlags::all(), caps);
    let info = validator.validate(&module).map_err(|err| {
        anyhow::anyhow!(
            "Naga validation failure after loop lowering for '{entry_point}':\n{}",
            err.emit_to_string(wgsl)
        )
    })?;

    let reemitted =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .with_context(|| format!("re-emitting WGSL after loop lowering for '{entry_point}'"))?;
    Ok(super::normalize_wgsl(&reemitted, entry_point))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_continuing_and_break_if_preserving_clean_wgsl() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn test_kernel_256(@builtin(local_invocation_index) lid: u32) {
    var rem: u32 = lid + 10u;
    var steps: u32 = 0u;
    var phi_sub: u32 = 4u;
    loop {
        let sub_cur = phi_sub;
        let cond = (sub_cur <= (rem >> 1u));
        var next_sub: u32 = 0u;
        if cond {
            next_sub = sub_cur << 1u;
            steps = steps + 1u;
        }
        continue;
        continuing {
            phi_sub = next_sub;
            break if !(cond);
        }
    }
    out_buf[lid] = steps;
}
"#;
        let lowered = lower_wgsl_loop_continuing(src, "test_kernel_256", false).unwrap();
        assert!(
            !lowered.contains("continuing"),
            "unexpected continuing in lowered WGSL:\n{lowered}"
        );
        assert!(
            !lowered.contains("break if"),
            "unexpected break if in lowered WGSL:\n{lowered}"
        );
        assert!(
            lowered.contains("\nfn test_kernel_256("),
            "entry point name was not restored:\n{lowered}"
        );
        let reparsed = naga::front::wgsl::parse_str(&lowered).unwrap();
        let mut validator =
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::empty());
        validator.validate(&reparsed).unwrap();
    }

    #[test]
    fn lowers_switch_continue_with_break_if_without_breaking_switch() {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn test_switch_256(@builtin(local_invocation_index) lid: u32) {
    var acc: u32 = 0u;
    var i: u32 = 0u;
    var next_i: u32 = 0u;
    loop {
        next_i = i + 1u;
        switch (i & 1u) {
            case 0u: {
                acc = acc + 3u;
                continue;
            }
            default: {
                acc = acc + 5u;
            }
        }
        continuing {
            i = next_i;
            break if i >= 4u;
        }
    }
    out_buf[lid] = acc;
}
"#;
        let lowered = lower_wgsl_loop_continuing(src, "test_switch_256", false).unwrap();
        assert!(
            !lowered.contains("continuing"),
            "unexpected continuing:\n{lowered}"
        );
        assert!(
            !lowered.contains("break if"),
            "unexpected break if:\n{lowered}"
        );
        assert!(
            lowered.contains("_loop_exit_"),
            "expected _loop_exit guard local for switch continue:\n{lowered}"
        );
        let reparsed = naga::front::wgsl::parse_str(&lowered).unwrap();
        for stmt in reparsed.entry_points[0].function.body.iter() {
            if let Statement::Loop { body, .. } = stmt {
                for inner in body.iter() {
                    if let Statement::Switch { cases, .. } = inner {
                        assert!(
                            !cases.iter().any(|c| {
                                c.body
                                    .iter()
                                    .any(|s| matches!(s, Statement::If { accept, .. } if matches!(accept.last(), Some(Statement::Break))))
                            }),
                            "break_if lowered to Switch-local Break instead of loop-exit flag:\n{lowered}"
                        );
                    }
                }
            }
        }
        let mut validator =
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::empty());
        validator.validate(&reparsed).unwrap();
    }
}
