/** Registers the harness mapping for `node --import`. */
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
register('./harness-loader.mjs', pathToFileURL(import.meta.filename));
