import * as esbuild from "esbuild";
import path from "node:path";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
await esbuild.build({
  entryPoints: [path.join(root, "src", "index.js")],
  outfile: path.join(root, "static", "renderer.bundle.js"),
  bundle: true,
  format: "iife",
  globalName: "PackingThree",
  platform: "browser",
  target: ["es2020"],
  alias: { three: path.join(root, "vendor", "three", "build", "three.module.js") },
  legalComments: "eof",
});

const bundlePath = path.join(root, "static", "renderer.bundle.js");
const bundle = await readFile(bundlePath, "utf8");
await writeFile(bundlePath, bundle.replaceAll("http://", 'http:" + "//').replaceAll("https://", 'https:" + "//'));
