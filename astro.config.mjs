// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	site: 'https://happy-llm-en.hoangdung24.workers.dev',
	markdown: {
		// Sätteri, Astro 7's default processor, has no remark/rehype hooks, so switch
		// to unified to get remark-math + rehype-katex for KaTeX math blocks.
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
	},
	build: {
		// Inline all page CSS into the HTML. The stylesheet is ~18KB of
		// render-blocking CSS on every page, so a request costs more than the bytes.
		inlineStylesheets: 'always',
	},
	integrations: [
		starlight({
			title: 'Happy-LLM',
			description: 'English translation of the Happy-LLM tutorial (datawhalechina/happy-llm)',
			customCss: ['./src/styles/custom.css'],
			components: {
				// Injects the font preload <link> into every page's <head>.
				Head: './src/components/Head.astro',
			},
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/datawhalechina/happy-llm' }],
			sidebar: [
				{
					label: 'Happy-LLM (English)',
					items: [
						{ label: 'Ch 1: NLP Fundamental Concepts', slug: 'chapter1-nlp-fundamental-concepts-en' },
						{ label: 'Ch 2: Transformer Architecture', slug: 'chapter2-transformer-architecture-en' },
						{ label: 'Ch 3: Pre-trained Language Models', slug: 'chapter3-pre-trained-language-models-en' },
						{ label: 'Ch 4: Large Language Models', slug: 'chapter4-large-language-models-en' },
						{ label: 'Ch 5: Build Your Own Large Model', slug: 'chapter5-build-your-own-large-model-en' },
						{ label: 'Ch 6: Training LLMs in Practice', slug: 'chapter6-training-llms-in-practice-en' },
						{ label: 'Ch 6 Supplement: Preference Alignment (WIP)', slug: 'chapter6-preference-alignment-wip-en' },
						{ label: 'Ch 7: LLM Applications', slug: 'chapter7-llm-applications-en' },
						{ label: 'Ch 8: LLM Reinforcement Learning', slug: 'chapter8-llm-reinforcement-learning-en' },
					],
				},
			],
		}),
		sitemap(),
	],
});
