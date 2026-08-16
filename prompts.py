"""
prompts.py - Prompt templates for the Loan Decision Support System (Lab 4)

How the prompts evolved:

Summarization:
V1 was just a one line prompt "Summarize this:", no role given, no constraints, no fixed
sentence count. Based on the output, it adds its own opinion and gives details not stated
in the letter, for example it said Kofi has no prior experience when the letter just said
he hasn't started the businesses yet. It also left out important details like Kwame having
no collateral. V2 fixed this by giving the model a role (assistant to a microfinance loan
officer) and constraints, be factual, neutral, don't invent details, and use exactly 3-4
sentences. V2 stayed factual and used the right number of sentences, so this is the version
used in the final system. Run at temperature=0.

Extraction:
Built with an explicit schema so the model knows exactly which keys to return, one few-shot
example that is NOT from the six letters (using a letter I wrote myself), because if the
example is one of the actual letters being processed, the model can just memorise it and
reproduce it instead of actually extracting, same idea as overfitting. Also added "if a
field is not stated in the letter, use null, do not guess" to stop the model from giving
plausible but false info for missing fields, which is hallucination. Run at temperature=0
because extraction has one correct answer per letter, unlike a creative task where you
want variation.

Brief:
Takes the letter and the extracted JSON together and produces 4 sections, strengths, risks,
missing information, and a suggested next step. The model is told not to output
"approve" or "reject". Practical reason: the model only sees what is in the letter, it
doesn't have access to follow up questions, interviews, or extra documents the loan officer
might get later, so letting it decide would be deciding on partial information. Ethical
reason: if the model made the decision, no one is really responsible for that decision the
way a human loan officer is responsible, and there is a risk of discrimination if the
training data has patterns like more loans historically granted to one group over another.
Run at temperature=0.
"""

SUMMARY_SYSTEM = (
    "You are an assistant to a microfinance loan officer in Ghana. "
    "Your job is to summarize loan application letters into short, factual briefs. "
    "Rules: be strictly factual and neutral; do not invent, assume, or infer any detail "
    "not explicitly stated in the letter; do not add opinions or recommendations; "
    "write exactly 3-4 sentences."
)

def SUMMARY_PROMPT(letter_text):
    return f"Summarize this loan application:\n\n{letter_text}"


EXTRACT_SYSTEM = (
    "You are a data extraction assistant for a microfinance loan officer. "
    "You extract structured facts from loan application letters. "
    "You must respond with ONLY a valid JSON object, no explanations, no markdown fences, "
    "no extra text before or after the JSON. "
    "If a field is not explicitly stated in the letter, use null. Do not guess or infer."
)

EXTRACT_PROMPT = """Extract the following fields from the loan application letter below and return them as a JSON object with EXACTLY these keys:

- applicant_name (string)
- amount_ghs (number)
- purpose (string)
- monthly_profit_ghs (number or null)
- has_collateral_or_guarantor (boolean)
- repayment_months (number or null)

If a field is not stated in the letter, use null. Do not guess.

Example:
"Dear Sir, My name is Ama Pokuaa, a snack vendor in Accra. I request GHS 5,000 to set up a vending space. My monthly profit is about GHS 600. My mother will guarantee the loan. I propose to repay GHS 300 monthly over 18 months."

JSON:
{{
  "applicant_name": "Ama Pokuaa",
  "amount_ghs": 5000,
  "purpose": "set up a vending space",
  "monthly_profit_ghs": 600,
  "has_collateral_or_guarantor": true,
  "repayment_months": 18
}}

Now extract from this letter:

{letter_text}

JSON:"""


BRIEF_SYSTEM = (
    "You are an assistant to a microfinance loan officer in Ghana. "
    "Your job is to produce a decision-support brief for a loan application, based ONLY on "
    "the letter and the extracted data provided. You must be factual and grounded — do not "
    "invent details, do not speculate beyond what is stated. "
    "IMPORTANT: You do not make loan decisions. Final approval or rejection is always made "
    "by a human loan officer. You must NEVER output the words 'approve' or 'reject', and you "
    "must never recommend that the loan be granted or denied. Your job ends at surfacing "
    "information and suggesting a procedural next step."
)

BRIEF_PROMPT = """Below is a loan application letter and structured data extracted from it.
Produce a decision-support brief with EXACTLY these four sections:

1. Strengths (bullet points, grounded only in the letter)
2. Risks / Red flags (bullet points)
3. Missing information the officer should request
4. Suggested next step (choose ONE of: "invite for interview", "request additional documents",
   "flag for senior review", "request guarantor/collateral details" — do NOT say "approve" or "reject")

Letter:
{letter_text}

Extracted data (JSON):
{extracted_json}

Brief:"""
