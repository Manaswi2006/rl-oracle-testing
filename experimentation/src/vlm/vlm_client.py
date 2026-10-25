"""Thin OpenAI-compatible client for a local vLLM server (Qwen3-VL).  No paid API is involved."""
import base64, math, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from openai import OpenAI

TASK_DESC = {"button": "to press the red button down completely from top to bottom",
             "hammer": "to hammer the grey nail completely in with a red hammer",
             "dial": "to turn the red line to the bottom of the dial",                       # unseen-task test (phrasing taken verbatim from CoRe's RRM/prompt.py)
             "peg": "to insert the green peg into the hole of the red bolck"}       # task phrasing from the CoRe repo (RRM/prompt.py)

# paraphrased outcome questions: the robot's task is given as an instruction only (no goal image, no state information)
OUTCOME_PROMPTS = [
    "A robot arm is working on this task: {task}. The {n} images are frames sampled in time order from ONE episode (first to last). Did the robot successfully complete the task during the episode? Answer with one word: Yes or No.",
    "Look at these {n} chronological frames of a robot episode. The goal of the robot is {task}. Is the goal achieved by the end of the episode (or at some point in it)? Reply Yes or No only.",
    "Task: the robot has to accomplish the following: {task}. You see {n} frames from the beginning to the end of an attempt. Judging only from the images, was the task fully accomplished? Answer Yes or No.",
    "You are a strict judge. The instruction for the robot was: {task}. Here are {n} time-ordered frames of its attempt. Has the instruction been completely carried out? Answer 'Yes' only if it clearly has, otherwise 'No'.",
    "Frames 1 to {n} show a robot attempt in time order. Instruction: {task}. Did the robot succeed? One word answer, Yes or No.",
]
PREF_PROMPT = ("A robot arm has to do this task: {task}. Below are two short clips, Clip A ({n} frames, in time order) and Clip B ({n} frames, in time order). "
               "Which clip shows MORE progress toward completing the task (a clip that is closer to the finished task is better)? Answer with one letter: A or B.")


def b64(jpg_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()


class VLM:
    def __init__(self, ports=(8000,), model="qwen3vl8b", workers=16):
        self.clients = [OpenAI(base_url=f"http://localhost:{p}/v1", api_key="none", timeout=300) for p in ports]
        self.model, self.workers, self._i = model, workers, 0

    def _chat(self, content, max_tokens=2, logprobs=True):
        c = self.clients[self._i % len(self.clients)]; self._i += 1
        kw = dict(model=self.model, messages=[{"role": "user", "content": content}], max_tokens=max_tokens, temperature=0.0)
        if logprobs:
            kw.update(logprobs=True, top_logprobs=8)
        for attempt in range(4):
            try:
                return c.chat.completions.create(**kw).choices[0]
            except TypeError:
                raise
            except Exception as e:
                time.sleep(2 + attempt * 3); err = e
        raise err

    @staticmethod
    def _p(choice, pos, neg):
        """P(pos token) / (P(pos)+P(neg)) from first-token top logprobs; also the greedy first token."""
        lp = choice.logprobs.content[0].top_logprobs
        ps = {t.token.strip().lower(): math.exp(t.logprob) for t in lp}
        a = ps.get(pos, 0.0); b = ps.get(neg, 0.0)
        return (a / (a + b) if a + b > 0 else 0.5), choice.logprobs.content[0].token.strip().lower()

    def outcome(self, frames_jpeg, task, prompt_idx):
        txt = OUTCOME_PROMPTS[prompt_idx].format(task=TASK_DESC[task], n=len(frames_jpeg))
        content = [{"type": "image_url", "image_url": {"url": b64(f)}} for f in frames_jpeg] + [{"type": "text", "text": txt}]
        ch = self._chat(content)
        p, tok = self._p(ch, "yes", "no")
        return dict(p_yes=p, vote=int(tok.startswith("yes")), token=tok)

    def preference(self, frames_a, frames_b, task):
        """returns P(A better) for the given order."""
        n = len(frames_a)
        txt = PREF_PROMPT.format(task=TASK_DESC[task], n=n)
        content = [{"type": "text", "text": "Clip A:"}] + [{"type": "image_url", "image_url": {"url": b64(f)}} for f in frames_a] + \
                  [{"type": "text", "text": "Clip B:"}] + [{"type": "image_url", "image_url": {"url": b64(f)}} for f in frames_b] + [{"type": "text", "text": txt}]
        ch = self._chat(content)
        p, tok = self._p(ch, "a", "b")
        return dict(p_a=p, token=tok)

    def pmap(self, fn, items):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(fn, items))


# ---- frozen outcome judge (selected on the dev split of the episode bank, same prompt for every task) -----------------------------------
COT_OUTCOME = [
    "The robot's instruction was: {task}. Image 1 is the first frame; images 2-5 are the last four frames in time order. First describe in two short sentences what is visible in the first and in the last frame "
    "(position of the objects relevant to the instruction), then decide whether the instruction has been completely carried out by the end. Finish with a final line exactly 'VERDICT: Yes' or 'VERDICT: No'.",
    "A robot was told: {task}. You are shown its starting frame (image 1) and the final four frames of the attempt (images 2-5, in time order). Briefly note how the relevant objects changed between the start and the end, "
    "then judge whether the instruction was fully accomplished. End with exactly one line: 'VERDICT: Yes' or 'VERDICT: No'.",
    "Instruction for the robot: {task}. Image 1 shows the scene before the attempt, images 2 to 5 show the end of the attempt. Compare the objects that matter for the instruction before and after, in two sentences, "
    "and then answer whether the task was completely done. Your last line must be 'VERDICT: Yes' or 'VERDICT: No'.",
]


def judge_outcome(vlm, frames16, task, k=3):
    """Frozen episode-outcome judge: first frame + last four frames, short reasoning, then a verdict; k paraphrases. Returns votes list and raw texts."""
    import re
    sel = [frames16[0]] + frames16[-4:]
    votes, texts = [], []
    for q in COT_OUTCOME[:k]:
        content = [{"type": "image_url", "image_url": {"url": b64(f)}} for f in sel] + [{"type": "text", "text": q.format(task=TASK_DESC[task])}]
        ch = vlm._chat(content, max_tokens=220, logprobs=False)
        txt = ch.message.content or ""
        m = re.findall(r"VERDICT:\s*(Yes|No)", txt, re.I)
        votes.append(int(bool(m) and m[-1].lower() == "yes")); texts.append(txt[-200:])
    return votes, texts
