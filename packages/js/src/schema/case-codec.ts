/**
 * Bijective snake_case wire JSON <-> camelCase TypeScript codec (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * Preserves user-supplied dictionary keys (`userMetadata`, `controls`, `controlsMeta`,
 * `tensorDirectory`, `conditionalMin`, `conditionalMax`, `conditionalPhysicalAnchors`,
 * `scaleLabels`, `optionLabels`) verbatim while translating all schema structural keys
 * bijectively and enforcing Rust `serde` `skip_serializing_if` rules.
 *
 * @module
 */

const SPECIAL_SNAKE_TO_CAMEL: Readonly<Record<string, string>> = {
  arena_nonce_96: "arenaNonce96",
  g_audio_uplo_u_f64: "gAudioUploUF64",
};

const SPECIAL_CAMEL_TO_SNAKE: Readonly<Record<string, string>> = {
  arenaNonce96: "arena_nonce_96",
  gAudioUploUF64: "g_audio_uplo_u_f64",
};

/**
 * Schema fields whose values are open `Record<string, V>` maps where keys are user/model data
 * rather than schema field names (and therefore must never have their keys case-converted).
 */
const USER_KEY_MAP_FIELDS = new Set<string>([
  "user_metadata",
  "userMetadata",
  "controls",
  "controls_meta",
  "controlsMeta",
  "conditional_min",
  "conditionalMin",
  "conditional_max",
  "conditionalMax",
  "conditional_physical_anchors",
  "conditionalPhysicalAnchors",
  "scale_labels",
  "scaleLabels",
  "option_labels",
  "optionLabels",
]);

/**
 * Schema fields whose keys are user/model identifiers (`tensor_directory`), while each value
 * is a nested schema struct (`TensorSliceEntry`) whose own keys ARE case-converted.
 */
const STRUCT_VALUE_MAP_FIELDS = new Set<string>(["tensor_directory", "tensorDirectory"]);

/**
 * Converts a snake_case schema key into its canonical camelCase TypeScript property name.
 */
export function snakeToCamelKey(key: string): string {
  const special = SPECIAL_SNAKE_TO_CAMEL[key];
  if (special !== undefined) {
    return special;
  }
  return key.replace(/_([a-z])/g, (_, ch: string) => ch.toUpperCase());
}

/**
 * Converts a camelCase TypeScript property name into its canonical snake_case wire JSON key.
 */
export function camelToSnakeKey(key: string): string {
  const special = SPECIAL_CAMEL_TO_SNAKE[key];
  if (special !== undefined) {
    return special;
  }
  return key.replace(/[A-Z]/g, (ch) => `_${ch.toLowerCase()}`);
}

/**
 * Recursively transforms a parsed wire JSON value from snake_case schema keys to camelCase.
 */
export function wireToCamelObject(value: unknown, parentField?: string): unknown {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => wireToCamelObject(item));
  }
  const record = value as Record<string, unknown>;
  if (parentField !== undefined && USER_KEY_MAP_FIELDS.has(parentField)) {
    const preserved: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      preserved[k] = v;
    }
    return preserved;
  }
  if (parentField !== undefined && STRUCT_VALUE_MAP_FIELDS.has(parentField)) {
    const mapped: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      mapped[k] = wireToCamelObject(v);
    }
    return mapped;
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(record)) {
    const camelKey = snakeToCamelKey(k);
    out[camelKey] = wireToCamelObject(v, camelKey);
  }
  return out;
}

/**
 * Recursively transforms a camelCase TypeScript schema object into snake_case wire JSON keys.
 */
export function camelToWireObject(value: unknown, parentField?: string): unknown {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => camelToWireObject(item));
  }
  const record = value as Record<string, unknown>;
  if (parentField !== undefined && USER_KEY_MAP_FIELDS.has(parentField)) {
    const preserved: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      preserved[k] = v;
    }
    return preserved;
  }
  if (parentField !== undefined && STRUCT_VALUE_MAP_FIELDS.has(parentField)) {
    const mapped: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(record)) {
      mapped[k] = camelToWireObject(v);
    }
    return mapped;
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(record)) {
    const snakeKey = camelToSnakeKey(k);
    out[snakeKey] = camelToWireObject(v, snakeKey);
  }
  return out;
}

function pruneNullField(rec: Record<string, unknown>, key: string): void {
  if (rec[key] === null || rec[key] === undefined) {
    rec[key] = undefined;
  }
}

function pruneMetadataWire(meta: unknown): void {
  if (meta === null || typeof meta !== "object") {
    return;
  }
  const metaRec = meta as Record<string, unknown>;
  pruneNullField(metaRec, "distribution_policy");
  const sigs = metaRec["signatures"];
  if (sigs === null || typeof sigs !== "object") {
    return;
  }
  const sigsRec = sigs as Record<string, unknown>;
  pruneNullField(sigsRec, "tier_arena_sha256");
  const rcpt = sigsRec["marketplace_receipt"];
  if (rcpt !== null && typeof rcpt === "object") {
    pruneNullField(rcpt as Record<string, unknown>, "buyer_tier_arena_sha256");
  }
}

function pruneCalibrationWire(calRec: Record<string, unknown>): void {
  pruneNullField(calRec, "lobatto_physical_anchors");
  const condAnchors = calRec["conditional_physical_anchors"];
  const emptyObj =
    typeof condAnchors === "object" &&
    condAnchors !== null &&
    Object.keys(condAnchors as Record<string, unknown>).length === 0;
  if (condAnchors === null || condAnchors === undefined || emptyObj) {
    calRec["conditional_physical_anchors"] = undefined;
  }
  pruneNullField(calRec, "tempo_sync_default_subdivision");
}

function pruneControlsWire(ctrls: unknown): void {
  if (!Array.isArray(ctrls)) {
    return;
  }
  for (const ctrl of ctrls) {
    if (ctrl === null || typeof ctrl !== "object") {
      continue;
    }
    const ctrlRec = ctrl as Record<string, unknown>;
    const cal = ctrlRec["calibration"];
    if (cal === null || cal === undefined) {
      ctrlRec["calibration"] = undefined;
    } else if (typeof cal === "object") {
      pruneCalibrationWire(cal as Record<string, unknown>);
    }
  }
}

function pruneTensorSliceEntry(sl: Record<string, unknown>): void {
  for (const optKey of ["packing", "feature_dim", "packed_len", "shape"]) {
    pruneNullField(sl, optKey);
  }
}

/**
 * Applies the exact Rust `serde` `skip_serializing_if` rules to a snake_case wire header object
 * so `encodeHeader` produces byte-identical JSON to `serde_json::to_string(&DnamJsonHeader)`.
 */
export function pruneRustSkipSerializingIf(raw: Record<string, unknown>): Record<string, unknown> {
  pruneMetadataWire(raw["metadata"]);
  pruneControlsWire(raw["controls_schema"]);
  const tdir = raw["tensor_directory"];
  if (tdir !== null && typeof tdir === "object") {
    for (const sl of Object.values(tdir as Record<string, unknown>)) {
      if (sl !== null && typeof sl === "object") {
        pruneTensorSliceEntry(sl as Record<string, unknown>);
      }
    }
  }
  return raw;
}
