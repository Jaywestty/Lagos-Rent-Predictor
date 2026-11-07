import { writeFileSync } from "node:fs";
import process from "node:process";

const base = process.env.VITE_API_BASE_URL || "http://localhost:8000";
const res = await fetch(`${base}/openapi.json`);
if (!res.ok) {
  console.error("OpenAPI fetch failed. Is backend running?");
  process.exit(1);
}
const spec = await res.json();
const schemas = spec.components?.schemas || {};
const out = `// Generated from OpenAPI. Do not edit by hand.\nexport type OpenAPI = ${JSON.stringify(Object.keys(schemas), null, 2)};\nexport const count = ${Object.keys(schemas).length};\n`;
writeFileSync("src/types/generated.ts", out);
