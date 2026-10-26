"""The thinking-effort surface: per-model levels, the instruction for templates that ignore the level, and off.

Run: python -m unittest serve.test_effort_modes -v
"""
import unittest

from serve.frontend import (ChatTemplate, effort_levels, openai_to_messages, with_effort_instruction)  # noqa: E402


class Levels(unittest.TestCase):
    def test_a_template_that_reads_enable_thinking_offers_off(self):
        self.assertEqual(effort_levels("{% if enable_thinking is false %}x{% endif %}", True),
                         ["off", "low", "medium", "high"])

    def test_a_model_that_always_thinks_offers_no_off(self):
        # Flint's template has neither switch: it cannot stop thinking, so Off must not be offered.
        self.assertEqual(effort_levels("<|im_start|>{{ messages }}", True), ["low", "medium", "high"])

    def test_no_system_turn_means_no_levels(self):
        # A template that cannot render a system role cannot carry the instruction either.
        self.assertEqual(effort_levels("no switches at all", False), [])


class Instruction(unittest.TestCase):
    def test_prepended_to_the_existing_system_turn(self):
        messages = [{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hi"}]
        out = with_effort_instruction(messages, "low")
        self.assertTrue(out[0]["content"].startswith("Reasoning effort is set to low."))
        self.assertTrue(out[0]["content"].endswith("Be brief."))
        self.assertEqual(out[1], messages[1])

    def test_a_system_turn_is_added_when_there_is_none(self):
        out = with_effort_instruction([{"role": "user", "content": "hi"}], "xhigh")
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("xhigh", out[0]["content"])
        self.assertEqual(out[1], {"role": "user", "content": "hi"})

    def test_off_and_unknown_levels_add_nothing(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertIs(with_effort_instruction(messages, "none"), messages)
        self.assertEqual(with_effort_instruction(messages, "none"), messages)

    def test_list_content_keeps_its_parts(self):
        messages = [{"role": "system", "content": [{"type": "text", "text": "Rules."}]},
                    {"role": "user", "content": "hi"}]
        out = with_effort_instruction(messages, "medium")
        self.assertEqual(out[0]["content"][0]["type"], "text")
        self.assertIn("Reasoning effort is set to medium", out[0]["content"][0]["text"])


class Request(unittest.TestCase):
    def test_a_top_level_enable_thinking_false_is_off(self):
        _messages, _tools, kwargs = openai_to_messages(
            {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False})
        self.assertEqual(kwargs, {"enable_thinking": False})

    def test_the_level_survives_as_kwargs(self):
        _messages, _tools, kwargs = openai_to_messages(
            {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"})
        self.assertEqual(kwargs, {"reasoning_effort": "xhigh"})

    def test_an_explicit_reasoning_effort_false_is_off(self):
        _messages, _tools, kwargs = openai_to_messages(
            {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": False})
        self.assertEqual(kwargs, {"enable_thinking": False})


class Templates(unittest.TestCase):
    """The template families this host serves, through ChatTemplate's own detection."""

    def _template(self, path):
        return ChatTemplate(path)

    def test_qwen38_default_template_reads_the_level(self):
        template = self._template(r"G:\PolyStrata\serve\chat_template.jinja")
        self.assertTrue(template.supports_reasoning_effort)
        self.assertTrue(template.supports_enable_thinking)

    def test_qwen36_template_does_not_read_the_level(self):
        template = self._template(r"G:\PolyStrata\packs\qwen36abl-q4_k_m\tokenizer\chat_template.jinja")
        self.assertFalse(template.supports_reasoning_effort)
        self.assertTrue(template.supports_enable_thinking)

    def test_flint_template_reads_neither(self):
        template = self._template(r"G:\PolyStrata\packs\flint-ud-q3_k_xl\tokenizer\chat_template.jinja")
        self.assertFalse(template.supports_reasoning_effort)
        self.assertFalse(template.supports_enable_thinking)


if __name__ == "__main__":
    unittest.main()
