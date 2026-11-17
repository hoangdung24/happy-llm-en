// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Happy-LLM',
			description: 'English translation of the Happy-LLM tutorial (datawhalechina/happy-llm)',
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
	],
});
