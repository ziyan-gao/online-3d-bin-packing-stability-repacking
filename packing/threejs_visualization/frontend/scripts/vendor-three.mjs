import { cp, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "node_modules", "three");
const target = path.join(root, "vendor", "three");

await rm(target, { recursive: true, force: true });
await mkdir(path.join(target, "build"), { recursive: true });
await mkdir(path.join(target, "examples", "jsm"), { recursive: true });
await mkdir(path.join(root, "static", "licenses"), { recursive: true });
await cp(path.join(source, "build", "three.module.js"), path.join(target, "build", "three.module.js"));
await cp(path.join(source, "examples", "jsm", "controls", "OrbitControls.js"), path.join(target, "examples", "jsm", "controls", "OrbitControls.js"), { recursive: true });
await cp(path.join(source, "examples", "jsm", "lines"), path.join(target, "examples", "jsm", "lines"), { recursive: true });
await cp(path.join(source, "LICENSE"), path.join(target, "LICENSE"));
await cp(path.join(source, "LICENSE"), path.join(root, "static", "licenses", "three-LICENSE.txt"));
