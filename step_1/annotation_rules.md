# Annotation Guidelines (5-Class Support/Response Taxonomy)

## Goal

Each response message must be assigned **exactly ONE label** from the following five mutually exclusive categories:

1. Practical / Actionable Advice
2. Warnings or Cautions
3. Emotional Support
4. Personal Opinions or Anecdotes
5. Appraisal (evaluation, judgment, or validation)

Annotate the **speaker's intent**, not the topic, not keywords, and not your interpretation of whether the advice is good.

---

## Unit of Annotation

The annotation unit is **one complete message/comment**.

* A message can be one sentence or multiple paragraphs.
* Label the **dominant communicative function** of the entire message.
* Do NOT label per sentence.

If a message contains multiple functions, apply the **dominance rule** (defined below).

---

## General Principles

### 1) Intent > Wording

Do not label based on specific words like *"sorry"*, *"should"*, *"I think"*.
You must decide **what the speaker is trying to do**.

Example:

> "I'm sorry this happened. You should talk to your advisor tomorrow."

Even though it contains empathy words, the **goal** is directing action → Advice.

---

### 2) Dominance Rule

If multiple categories appear, choose the label that represents the **primary purpose** of the message.

Use this priority order when unclear:

**Advice > Warning > Emotional Support > Personal Anecdote > Appraisal**

Reason: some categories are functionally stronger (they try to change behavior).

---

### 3) Ignore Politeness Wrappers

Many messages begin with emotional phrases but serve another function.

Example:

> "I'm sorry you're going through this, but you need to stop contacting them."

→ Advice (not Emotional Support)

---

# Category Definitions and Rules

---

## 1. Practical / Actionable Advice

### Definition

The speaker **recommends a specific action** the recipient should take to solve or improve a situation.

This includes:

* instructions
* suggestions
* strategies
* problem-solving steps
* coping techniques

### Key Test

**If the reader could follow steps after reading the message → Advice**

### Strong Indicators

* "You should…"
* "Try…"
* "I recommend…"
* "The best thing to do is…"
* "Go talk to…"
* "Start by…"

### Examples (POSITIVE)

* "You should email your professor and explain the situation."
* "Try setting a 10-minute timer and just start the assignment."
* "Block them on social media."
* "Make a study schedule and break the work into chunks."

### Borderline Cases

**Advice + empathy → Advice**

> "I know this is hard. You should probably see a counselor."

### NOT Advice

General encouragement without an action:

> "Things will get better." → Emotional Support

---

## 2. Warnings or Cautions

### Definition

The speaker predicts **negative consequences** or alerts the reader to danger or risk.

Purpose: **prevent harm**, not solve the problem.

### Key Test

Is the message mainly saying:
**"Something bad may happen if you continue."**

### Examples (POSITIVE)

* "If you keep ignoring this, you could fail the course."
* "Be careful — they might be manipulating you."
* "That sounds like a scam."
* "You could get in trouble for that."

### Important Distinction

Warning ≠ Advice

| Message                                       | Label   |
| --------------------------------------------- | ------- |
| "You should leave that job."                  | Advice  |
| "If you stay, they will keep exploiting you." | Warning |

---

## 3. Emotional Support

### Definition

The speaker's primary purpose is to **comfort, reassure, empathize, or emotionally soothe** the recipient.

No attempt to solve the problem.

### Key Test

Remove the message — would the person only lose comfort, not guidance?

### Typical Functions

* empathy
* reassurance
* validation of feelings
* compassion

### Examples (POSITIVE)

* "I'm really sorry you're going through this."
* "You're not alone."
* "That sounds incredibly hard."
* "I'm here for you."

### Important Rule

If **no action is suggested**, it is Emotional Support even if it sounds helpful.

### NOT Emotional Support

If action appears:

> "I'm sorry. You should talk to HR." → Advice

---

## 4. Personal Opinions or Anecdotes

### Definition

The speaker talks about **their own experiences, beliefs, or story** without primarily trying to help or guide the reader.

The focus is the **speaker**, not the recipient.

### Key Test

Could the message exist even if the recipient had never posted?

If yes → anecdote/opinion.

### Examples (POSITIVE)

* "This happened to me in my second year."
* "I also failed calculus once."
* "I personally hate group projects."
* "When I was in that situation, I just ignored it."

### Important Rule

If the anecdote leads to advice → Advice.

Example:

> "I went through this too. You should talk to a therapist."
> → Advice

---

## 5. Appraisal (Evaluation / Judgment / Validation)

### Definition

The speaker **evaluates a person, behavior, decision, or situation**.

Not comfort. Not instruction. Not warning.

It is an **assessment**.

### Key Test

Is the message mainly telling the reader **how to interpret the situation**?

### Examples (POSITIVE)

* "You did the right thing."
* "That was irresponsible."
* "Your professor is being unfair."
* "Your feelings are completely valid."

### Important Distinction

Validation ≠ Emotional Support

| Message                                  | Label             |
| ---------------------------------------- | ----------------- |
| "I'm sorry you feel this way."           | Emotional Support |
| "You are justified in feeling this way." | Appraisal         |

---

# Decision Procedure (Annotator Checklist)

Follow this order:

1. Does the message tell the reader to DO something?
   → Advice

2. Does it predict harm or danger?
   → Warning

3. Is it primarily comforting?
   → Emotional Support

4. Is it mainly about the speaker's story or belief?
   → Personal Anecdote

5. Is it judging/evaluating behavior?
   → Appraisal

---

# Special Cases

### Questions

Questions that imply action count as Advice:

> "Why don't you talk to your advisor?" → Advice

Neutral questions:

> "What happened?" → Not annotatable (skip or mark other depending on dataset rules)

---

### Mixed Messages

Label based on the **final communicative effect**, not the opening.

> "I'm sorry this happened. You need to report them immediately."
> → Advice

---

### Sarcasm

Annotate intended meaning, not literal wording.

> "Yeah great idea, keep texting your ex at 2am."
> → Warning/Caution

---

# Summary Table

| Category          | Purpose                   |
| ----------------- | ------------------------- |
| Advice            | Change behavior           |
| Warning           | Prevent harm              |
| Emotional Support | Provide comfort           |
| Anecdote          | Share personal experience |
| Appraisal         | Provide evaluation        |
