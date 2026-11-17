# Happy-LLM (English Translation)

An English translation of [Happy-LLM](https://github.com/datawhalechina/happy-llm), a systematic tutorial on large language models originally written in Chinese by [Datawhale](https://github.com/datawhalechina).

## Contents

| Chapter | File | Topic |
|---|---|---|
| 1 | [`chapter1-nlp-fundamental-concepts-en.md`](chapter1-nlp-fundamental-concepts-en.md) | NLP fundamental concepts |
| 2 | [`chapter2-transformer-architecture-en.md`](chapter2-transformer-architecture-en.md) | Transformer architecture |
| 3 | [`chapter3-pre-trained-language-models-en.md`](chapter3-pre-trained-language-models-en.md) | Pre-trained language models |
| 4 | [`chapter4-large-language-models-en.md`](chapter4-large-language-models-en.md) | Large language models |
| 5 | [`chapter5-build-your-own-large-model-en.md`](chapter5-build-your-own-large-model-en.md) | Build your own large model |
| 6 | [`chapter6-training-llms-in-practice-en.md`](chapter6-training-llms-in-practice-en.md) | Training LLMs in practice |
| 6+ | [`chapter6-preference-alignment-wip-en.md`](chapter6-preference-alignment-wip-en.md) | Preference alignment supplement (WIP in the original) |
| 7 | [`chapter7-llm-applications-en.md`](chapter7-llm-applications-en.md) | LLM applications |
| 8 | [`chapter8-llm-reinforcement-learning-en.md`](chapter8-llm-reinforcement-learning-en.md) | LLM reinforcement learning |

## Website

The `site/` directory contains an [Astro](https://astro.build) + [Starlight](https://starlight.astro.build) documentation site that publishes these chapters online, deployed to Cloudflare Workers (static assets).

```bash
cd site
npm install
npm run dev     # local dev server
npm run build   # build to dist/
npx wrangler deploy
```

## Notes on the translation

- All prose was translated from Chinese to English.
- Code blocks are unchanged from the original; Chinese comments inside code were translated to English.
- Chinese example strings inside code blocks are kept verbatim with English glosses in parentheses.
- Reference citations are kept verbatim, including Chinese-language sources.
- Translation was produced with AI assistance.

## License

The original Happy-LLM tutorial is licensed under CC BY-NC-SA 4.0 by Datawhale. This translation follows the same license.
