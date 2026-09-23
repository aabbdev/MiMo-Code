/**
 * Prompts for the RLM loop, adapted from Appendix C of arXiv 2512.24601.
 *
 * Three deliberate departures from the paper's text:
 *
 *  - The REPL language is JavaScript, not Python. The kernel is a bare QuickJS
 *    realm (no filesystem, no network, no imports), so examples that say
 *    `context[:10000]` or `re.split` would be wrong on both counts.
 *  - The sub-call warning the paper only adds for Qwen3-Coder is included for
 *    every model. Their finding was that without it a model "will try to perform
 *    a subcall on everything, leading to thousands of LM subcalls for basic
 *    tasks"; our budget would abort such a run instead of absorbing it.
 *  - GROUNDING IS AN OBLIGATION, not the paper's suggestion. The echo back to
 *    the root model is truncated, so printing a payload is not reading it. A
 *    measured run over a 2.3 MB tree answered with a confident description of
 *    all 119 files having observed 0.65 % of the bytes and made zero sub-calls —
 *    the descriptions were rewritten filenames. The text below says plainly that
 *    such an answer is an invention.
 */
import type { ModelMessage } from "ai"

export type ContextMeta = {
  type: string
  length: number
  prefix: string
  /** How many units `context_parts` holds — one per file, or `path#2` when a
   * large file was split further. The paper passes chunk lengths for the same
   * reason: the first decomposition attempt is what decides the run, and the
   * model cannot plan one without a unit of work. */
  partCount: number
}

const EXAMPLES = `
Example — read ONE part's content by routing it through a sub-LLM. Never describe
a part you have not sent to one:

\`\`\`repl
const part = context_parts[0];
const note = await llm_query(
  "In one line, what is this file responsible for? Cite an identifier that appears in it.\\n\\n" + part
);
console.log(context_part_names[0] + ": " + note);
\`\`\`

Example — the workhorse. PACK SEVERAL PARTS INTO EACH SUB-CALL: the budget counts
CALLS, and a sub-LLM can take far more than one file. This is how you read a whole
tree without exhausting it:

\`\`\`repl
const perCall = 150000;
const groups = [];
let current = { text: "", names: [] };
for (let i = 0; i < context_parts.length; i++) {
  if (current.text.length + context_parts[i].length > perCall && current.names.length > 0) {
    groups.push(current);
    current = { text: "", names: [] };
  }
  current.text += "### " + context_part_names[i] + "\\n" + context_parts[i] + "\\n\\n";
  current.names.push(context_part_names[i]);
}
if (current.names.length > 0) groups.push(current);

globalThis.notes = [];
for (const group of groups) {
  notes.push(await llm_query(
    "For EACH file delimited by a ### heading below, give one line saying what it is responsible for, " +
    "citing an identifier that actually appears in it. Start each line with the file name.\\n\\n" + group.text
  ));
}
console.log(notes.length + " sub-calls covered " + context_parts.length + " parts");
\`\`\`

Example — a question about the payload as a whole (counting, structure, cross-file
search) is the case for \`context\` and for code, not for sub-calls:

\`\`\`repl
const callers = context.split("\\n").filter((line) => line.includes("createBinaryImage"));
console.log(callers.length + " mentions");
\`\`\`
`

export function systemPrompt(meta: ContextMeta): string {
  return `You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

Your context is a ${meta.type} with ${meta.length} total characters, split into ${meta.partCount} parts. It begins with:

${meta.prefix}

The REPL environment is initialized with:

1. A \`context\` variable holding the whole payload as one string. Use it for questions about the payload as a whole — counting, structure, searching across everything.

2. A \`context_parts\` variable: an array of ${meta.partCount} strings, one per unit of work (one file, or \`name#2\` when a large file was split further), with a matching \`context_part_names\` array of names. THIS is your unit for reading content, and the natural input to a sub-call.

3. A \`query\` variable holding the question you must answer. Put it in your sub-call prompts rather than paraphrasing it — a sub-LLM that never sees the original question can only guess at it.

4. A \`llm_query\` function that allows you to query an LLM inside your REPL environment. It takes a prompt string and returns the answer.

5. A \`llm_query_batched\` function that takes an array of prompt strings and returns an array of answers, running them together. Prefer it over looping over \`llm_query\`.

6. The ability to use \`console.log()\` statements to view the output of your REPL code and continue your reasoning.

The REPL is JavaScript running in an isolated engine. There is no filesystem, no network, no \`require\`/\`import\`, and no timers. Work with \`context\` and \`context_parts\` and with values you compute; that is all you need.

IMPORTANT — WHERE YOUR ANSWER MAY COME FROM. What you print is returned to you TRUNCATED to a short prefix. Printing a file therefore does not put it in front of you, and it does not count as reading it. Saying what a part contains without having passed that part to a sub-LLM is not a shortcut: it is an invention, and a confident invented description is worse than an admitted gap. Code can tell you a file's size, name, length, and structure; only \`llm_query\` can tell you what it MEANS. Route every question of meaning through the sub-LLMs and build your answer out of what they return.

SCREEN BEFORE YOU READ. \`screen(question)\` makes one cheap pass over short previews of every part and returns the indices worth reading. On a large payload that is a fraction of the cost of reading it, so call it FIRST and read only what it returns — reading everything to answer one question is the expensive mistake. If the screening comes back empty or wrong, screen again with a differently worded question rather than falling back to reading the whole payload.

For a follow-up question about a part you have already asked about, use \`ask_about(index, question)\` instead of \`llm_query\`: it puts the part first and your question last, so the provider serves the part from its cache rather than charging for it again.

CONFIDENCE, REQUIRED EVERY STEP. End each reply with a single JSON object on its own line, reporting how confident you are in the step you are about to take:

{"confidence": 87}

A number in (0, 100]. Be precise and nuanced rather than habitual: it should say how sure you are that this step is the right one and that what you have established so far is correct. A value near 100 means you would be surprised to be wrong; a low value is useful information and costs you nothing.

VERDICT, INSIDE YOUR FINAL ANSWER. \`FINAL(...)\` carries two parts separated by three vertical bars: your conclusion in ONE short self-contained line, then the full answer.

  FINAL(the conclusion in one line ||| the complete answer, in as much detail as you need)

The first part is compared word-for-word against other attempts at this same question, so say the same thing in the same words, keep it under about 200 characters, and name the file or symbol your conclusion rests on — that makes it checkable against the payload rather than merely comparable to another opinion. The second part is what a reader reads, and it is where the detail belongs.

Make sure to explicitly look through the entire context in the REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub-LLMs are powerful -- they can fit a lot of characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute code in the REPL environment, wrap it in triple backticks with 'repl' language identifier:
${EXAMPLES}
IMPORTANT: Be very careful about using \`llm_query\` as it incurs high runtime costs and there is a hard budget on the number of sub-calls. Always batch as much information as reasonably possible into each call. For example, if you have 1000 lines of information to process, it is much better to split them into chunks of 5 and call \`llm_query\` on each chunk than to make 1000 individual calls. Minimize the number of sub-calls by batching related information together.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:

1. Use \`FINAL(your final answer here)\` to provide the answer directly
2. Use \`FINAL_VAR(variable_name)\` to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.`
}

export function userTurn(query: string): string {
  return `Query: ${query}`
}

/** Sent when the model neither returned a FINAL answer nor wrote runnable code,
 * which the paper's ablation shows is a common early stall ("I will do this"). */
export const NUDGE =
  "No code block was found in your reply. Reply with a ```repl code block to run, or with FINAL(...) / FINAL_VAR(...) if you have completed the task. Do not describe what you plan to do — run it."

/**
 * Re-stated after every step, because the requirement decays.
 *
 * Measured with DeepSeek-V4.1-Flash: the model emits its `{"confidence": ν}` on
 * the first reply and stops emitting it on the next one, once the history has
 * filled with code blocks and REPL echoes it is pattern-matching against. A rule
 * stated once at the top of a long prompt is a rule the trajectory forgets — and
 * without the report, selection loses its semantic signal and falls back to
 * reasoning length, which the paper's own ablation calls unreliable.
 */
export const CONFIDENCE_REMINDER = `[step complete] End your next reply with the required {"confidence": N} line.`

/**
 * The verdict is asked for once, at the very end, so it decays exactly like the
 * confidence line did — and it decays worse, because unlike confidence it has no
 * per-step occasion to be repeated. Measured TWICE: a 3-trajectory run reported
 * ZERO verdicts, and then a run that answered on iteration 10 of 12 reported none
 * either — because a reminder riding only the FINAL TURN note (which needs
 * `left <= 2`) is never seen by a trajectory that finishes early. It therefore
 * rides on every echo, like the confidence line.
 */
export const VERDICT_REMINDER = `[when you finish] FINAL(one-line verdict ||| the full answer).`

export type { ModelMessage }
