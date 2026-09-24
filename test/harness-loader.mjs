/**
 * Loader hook: map the harness-provided specifiers onto the installed copy.
 *
 * The adapter imports `@deepseek-ai/dsh-llm` / `@deepseek-ai/dsh-timeout`,
 * which only exist inside a harness installation. Mapping them here (instead
 * of rewriting the adapter's imports) keeps the tested file byte-identical to
 * the one that ships.
 *
 * resolve() must yield a URL, not a bare path.
 */
import { pathToFileURL } from 'node:url';

const HARNESS = process.env.DSH_HARNESS_DIR
  ?? '/root/.nvm/versions/node/v24.20.0/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai';

const MAP = {
  '@deepseek-ai/dsh-llm': `${HARNESS}/dsh-llm/lib/index.js`,
  '@deepseek-ai/dsh-timeout': `${HARNESS}/dsh-timeout/lib/index.js`,
  '@deepseek-ai/dsh-attachment': `${HARNESS}/dsh-attachment/lib/index.js`,
};

export async function resolve(specifier, context, next) {
  const mapped = MAP[specifier];
  if (mapped !== undefined) return { url: pathToFileURL(mapped).href, shortCircuit: true };
  return next(specifier, context);
}
