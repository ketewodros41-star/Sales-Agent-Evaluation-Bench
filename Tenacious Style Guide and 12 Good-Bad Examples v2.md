

Tenacious Style Guide v2
Tone markers, formatting rules, and 24 labeled outreach drafts
## Purpose
This guide codifies the Tenacious voice and provides 12 labeled "good" and 12 labeled "bad" outreach
drafts. The labeled drafts are training data for the agent's tone-preservation check and reference
material for any team member writing email or LinkedIn outreach in the Tenacious voice.
The drafts are anchored to real Tenacious context: the four ICP segments (recently-funded Series A/B
startups, mid-market platforms restructuring cost, engineering-leadership transitions, specialized
capability gaps), the four hiring signals (funding event, job-post velocity, layoffs, leadership change), the
AI maturity score (0–3), the four-stack bench composition (Python, Go, data, ML, infra), and the public
pricing bands.
## The Five Tone Markers

Every outreach, every reply, every discovery-call context brief must preserve these markers. A draft that
scores below 4/5 on any marker is regenerated. A draft that fails two or more markers is a brand
violation.
## 1. Direct
Clear, brief, actionable. No filler. Subject lines state intent — "Request," "Follow-up," "Context,"
"Question" — never "Quick," "Just," or "Hey." Body of cold outreach: 120 words maximum. One clear ask
per message.
## 2. Grounded
Every claim is supported by the hiring signal brief or the competitor gap brief. When a signal is weak
(fewer than five open roles, single low-confidence input), the agent asks rather than asserts. "Three open
Python roles since January — is hiring velocity matching the runway?" not "You're scaling aggressively."
## 3. Honest
Refuses claims that cannot be grounded in data. Never claims "aggressive hiring" if the job-post signal is
weak. Never over-commits bench capacity that bench_summary.json does not show. Never fabricates
peer-company practices to make a competitor gap brief look sharper. When a signal is missing, names
the absence and asks.
## 4. Professional

Language appropriate for founders, CTOs, and VPs of Engineering. Avoids internal jargon — "bench"
reads as offshore-vendor language to a prospect. Use "engineering team," "available capacity," or
"engineers ready to deploy." Never uses offshore clichés ("top talent," "world-class," "A-players,"
## "rockstar," "ninja").
## 5. Non-condescending
Frames competitor gaps as research findings or questions worth asking, never as failures of the
prospect's leadership. Senior engineering leaders know their own gaps; they do not need a cold email to
point them out. The value is the specificity of what peers are doing, not the implication that the prospect
is behind.
## Formatting Constraints
- Body length: Cold outreach max 120 words. Warm reply max 200 words. Re-engagement max 100
words.
- One ask per message. No stacking "and also would love to discuss X, Y, Z."
- Subject line under 60 characters. Gmail truncates above this on mobile.
- No emojis in cold outreach. Permitted in warm replies if the prospect sets the tone first.
- No marketing taglines in signature. Name, title, Tenacious, gettenacious.com. Nothing else.
- No attached PDFs in cold outreach. Send links inline only after the prospect has expressed
interest.
- Re-engagement emails carry new content, not guilt. No "following up again" or "circling back."
Add a new signal, a new question, or a new data point.
## Signature Template
[First name]
[Title — "Research Partner," "Delivery Lead," "Engagement Manager"]
## Tenacious Intelligence Corporation
gettenacious.com
## Channel Rules
Channel When to use
Email (primary) Cold outreach. Warm replies. Proposal and contract follow-up. The default
channel for every Tenacious prospect.

LinkedIn DM Permitted only when the prospect's email is unavailable, when the signal is
recent enough that LinkedIn timing matters (a leadership change
announcement within seven days), or when the prospect has engaged with
Tenacious content in the last 14 days. Never used as the first touch for a
Series A/B founder unless email has bounced or returned no contact.
SMS Only after the prospect has replied at least once and confirmed SMS is
acceptable for scheduling. Never as a cold first touch.
Voice Discovery call only, booked through Cal.com, delivered by a human Tenacious
delivery lead. The agent never voice-calls a prospect.
## Pre-flight Checklist Before Any Outreach Goes Out
The agent runs through this list before sending. A failed item triggers regeneration or human review.
Check Pass condition Fail behavior
Hiring signal
grounding
At least one signal from the
brief is referenced
specifically (named funding
amount and date, named
role count and trend,
named layoff event, named
leadership change with
date).
Regenerate. A draft with no specific signal is a tone
failure on Grounded.
## Confidence-awar
e phrasing
If the signal confidence in
the brief is "Medium" or
"Low," the draft uses
interrogative or conditional
language ("is hiring velocity
matching the runway?", "if
you're scoping additional
capacity"). If confidence is
"High," assertive language
is permitted.
Regenerate. Asserting on weak signal is a tone
failure on Honest.
## Bench-vs-engine
ering-team
language
The word "bench" does not
appear in any
prospect-facing message.
Acceptable substitutes:
Regenerate. "Bench" externally is a tone failure on
## Professional.

"engineering team,"
"available capacity,"
"engineers ready to
deploy."
## Bench-to-brief
match
If the draft commits to
capacity (a number of
engineers in a stack, a start
date), the commitment is
supported by
bench_summary.json. If
the brief asks for capacity
that the bench does not
show, route to human, do
not commit.
Replace commitment with a discovery-call routing
line.
Pricing scope If the draft mentions price,
it is from the public
quotable bands in
pricing_sheet.md (junior
monthly floor, mid-tier
monthly, starter project
floor, hourly blended).
Total contract values for
multi-phase engagements
are not invented.
Replace specific number with band, route to
discovery.
Word count
Cold body ≤ 120 words.
Warm reply ≤ 200 words.
## Re-engagement ≤ 100
words. Subject ≤ 60
characters.
Trim. If trimming requires removing the signal
grounding, regenerate.
One ask Body has one explicit call
to action. Common asks:
15-minute scoping call,
reply with current
bottleneck, send PDF
resource, no-action FYI.
Pick the highest-value ask, drop the others,
regenerate.

Banned phrase
scan
None of the banned
phrases (see list below)
appear in subject or body.
## Regenerate.
LinkedIn-roast
test
If the email were
screenshot and posted on
LinkedIn with the
prospect's annotation,
would Tenacious be
roasted? If yes, regenerate.
## Regenerate.
## Banned Phrases
These phrases do not appear in any Tenacious outreach. The agent's banned-phrase regex check fails the
draft if it sees any of them.
Phrase Why banned
world-class Marketing filler, unfalsifiable.
top talent Offshore-vendor cliché, signals low quality.
A-players Same as above.
rockstar / ninja / wizard Outdated vendor jargon.
skyrocket / supercharge / 10x Aggressive growth promises with no
substantiation.
I hope this email finds you well Generic, signals template.
just following up / circling back Re-engagement filler with no new content.
Quick question / Quick chat "Quick" implies the recipient's time is owed.
synergize / synergy / leverage / ecosystem Consultant jargon.
game-changer / disruptor / paradigm shift Hype with no substance.
our proprietary [X] system / our AI-powered [X] Black-box claims that invite skepticism.
You'll regret missing this / Don't miss out Fake urgency.
Per my last email Passive-aggressive.

our 500 employees / our 20 years of experience Self-centered, irrelevant to the prospect's signal.
I'll keep this brief — but [long paragraph] Performative concision that fails.
I noticed you're a [job title] Generic — the title alone is not a signal.
Twelve Good Drafts (Labeled)
Each draft is grounded in at least one named hiring signal and follows all five tone markers. The
metadata below each draft names which segment, which signal, and which Tenacious service line the
draft fits.

GOOD #1 — Series A funding + role velocity, high signal confidence
Subject Request: 15 minutes on your Q3 Python hiring
## Body Hi Maya,
You closed your $14M Series A in February and your open Python
engineering roles went from 2 to 7 in the last 60 days. The typical
bottleneck for teams in that state is recruiting capacity, not budget.
We place dedicated Python and data engineers, managed by Tenacious,
with a minimum three hours of synchronous overlap. We can plug a
team in within 48 hours while you continue your full-time search.
Would 15 minutes next week be useful? I'll bring two case studies from
Series A SaaS clients who hit the same wall.
## Best,
## Yabi
## Research Partner, Tenacious Intelligence Corporation
gettenacious.com
Why it works
- Names the funding amount and date, and the exact role-count
trend. Both are verifiable.
- Frames the gap as a typical pattern, not a personal failing of the
prospect.
- One ask: 15 minutes. No stacking.
- Avoids the word "bench" — uses "team" and "plug a team in."
- Names a concrete value-add for the call (two relevant case
studies), not a vague "discuss synergies."

- Body 89 words, subject 39 characters.
Signal grounding ICP Segment 1 (recently-funded Series A/B startups). Primary signal:
funding event, 120 days ago. Secondary: hiring velocity tripled.
Confidence: High on both.
GOOD #2 — Post-layoff cost-pressure pitch, mid-market restructuring
Subject Context: lower-cost engineering capacity post-restructure
## Body Hi Daniel,
I saw the announcement that your team contracted by about 12% in
March. Companies in your stage often need to maintain delivery output
while reducing fully-loaded cost — that is the engagement pattern we
run most often.
Tenacious places managed engineering teams under our project
management. Senior engineers in Python, data, and ML start from
$X,XXX/month, with a one-month minimum and two-week extension
blocks. No long-term commitment.
If you are scoping the next twelve months of delivery capacity, I can
share two short case studies from mid-market clients who replaced a
portion of their delivery cost this way.
## Best,
## Yabi
## Research Partner, Tenacious Intelligence Corporation
gettenacious.com
Why it works
- Acknowledges the layoff specifically and respectfully — names it
as "contraction," not as a failure.
- Frames the pitch as a pattern Tenacious sees often, removing
personal sting.
- Quotes only public-tier pricing bands and the 1-month minimum
from pricing_sheet.md.
- Conditional language: "If you are scoping..." — does not assume
need.
- Body 100 words. Avoids "bench," "top talent," and any urgency
framing.

Signal grounding ICP Segment 2 (mid-market platforms restructuring cost). Primary
signal: layoffs.fyi, 45 days ago, 12% headcount cut. Secondary:
post-layoff hiring posts indicating skill rebalance. Confidence: High.
GOOD #3 — New CTO 90-day vendor reassessment window
Subject Context: a brief on offshore engineering models
## Body Hi Priya,
Welcome to your new role at Helix — I saw the announcement on the
14th. New engineering leaders typically reassess vendor and offshore
mix in their first 90 days.
I do not want to add to your inbox in week three of a new job. I will
leave you with one thing: a one-page brief on the four offshore
engagement models we see most often, with the trade-offs honestly
laid out (including where each model fails).
If a 15-minute conversation in November would be useful, the calendar
is at gettenacious.com/yabi. If not, no follow-up.
## Best,
## Yabi
## Research Partner, Tenacious Intelligence Corporation
gettenacious.com
Why it works
- Names the leadership-change announcement date specifically.
- Self-aware: acknowledges it is a noisy moment, lowers the ask.
- Offers a real value-add (one-page brief with honest trade-offs
including failures) rather than a generic case study.
- Explicit "if not, no follow-up" — shows respect for time.
- Body 96 words. No banned phrases. No assumed need.
Signal grounding ICP Segment 3 (engineering-leadership transition). Primary signal: new
CTO announcement, 18 days ago. Secondary: none required — the
leadership change alone is the buying-window signal. Confidence: High.
GOOD #4 — Capability gap, AI maturity score 2 with high confidence
Subject Question: your MLOps function in 2026
## Body Hi Felix,

Three companies adjacent to yours in the loyalty-platform space — A,
B, and C — posted senior MLOps engineer roles in the last 90 days. Your
team has not, at least not publicly. Two readings: a deliberate choice, or
a function that has not yet been scoped.
We staff specialized squads (ML platform, agentic systems, data
contracts) on fixed-scope project engagements, typically 3 to 4 months.
Starter scopes from $XX,XXX. We do not pitch this where there is no
real need.
If you have already scoped this and decided against it, I would
genuinely be curious why — that is useful intelligence for us. If not, 15
minutes is enough to walk through what those three peer companies
are doing.
## Best,
## Yabi
## Research Partner, Tenacious Intelligence Corporation
gettenacious.com
Why it works
- Frames the gap as two readings, not a deficiency. Senior leaders
respect being given the benefit of the doubt.
- Asks for the prospect's reasoning if they have already decided —
positions Tenacious as a learner, not a vendor.
- References specific peer companies (would be named in the
actual brief) rather than "top quartile."
- Quotes only the public starter project floor.
- Body 117 words. One ask: 15-minute walk-through if interested.
Signal grounding ICP Segment 4 (specialized capability gaps). Primary signal: competitor
gap brief — three named peers with public MLOps openings; prospect
has none. AI maturity score: 2. Signal confidence: High (multiple peer
signals).
GOOD #5 — Weak signal, asks rather than asserts
Subject Question: are your data engineering hires keeping up?
## Body Hi Tom,
Two open data engineer roles on your careers page — I cannot tell from
the outside whether that means hiring is keeping pace or whether the
queue is longer than the postings suggest.

We place managed data and Python engineering teams, three-hour
overlap with US time zones, one-month minimum. If the queue is
longer than the posts, that is the pattern we solve most often.
If two roles is the actual demand and you are well-staffed to meet it,
ignore this. If the real number is higher, a 15-minute conversation costs
you nothing and gives me a chance to learn what you are seeing.
## Best,
## Yabi
## Research Partner, Tenacious Intelligence Corporation
gettenacious.com
Why it works
- Honestly names what the agent cannot tell from public signal —
"I cannot tell from the outside."
- Conditional framing: "If the queue is longer..." — does not assert
a need.
- Explicit out: "If two roles is the actual demand, ignore this."
Reduces friction; the prospect feels respected.
- Body 105 words. Subject is interrogative, matches the weak-signal
posture.
Signal grounding ICP Segment 1 or 2. Primary signal: 2 open roles, ambiguous.
Confidence: Low. Bench match (Python, data): supported by
bench_summary.json. The agent correctly chose interrogative phrasing
for low-confidence signal.
GOOD #6 — Resource value-add, no-pitch first touch
Subject Resource: Series A engineering scale-up checklist
## Body Hi Ana,
You closed your seed extension in October and your first three
engineering hires are public on LinkedIn. The window between now and
your Series A is the one where most teams' delivery process either
compounds or stalls.
I put together a one-page checklist of the seven decisions that
determine which side a team lands on (when to introduce code review
formality, when to write the first runbook, when offshore augmentation
pays back, when it does not). Two of the items are arguments against
hiring an outsourced team in your stage.

Want me to send the PDF? No follow-up if you are not interested.
## Best,
## Yabi
## Research Partner, Tenacious Intelligence Corporation
gettenacious.com
Why it works
- Pure value-add first touch — no pitch, no calendar ask.
- Names two items that argue against Tenacious's own service —
earns trust by demonstrating honesty.
- References specific public signals (seed extension, three hires).
- One ask: "Want me to send the PDF?" Lowest possible friction.
- Body 116 words. No banned phrases.
Signal grounding ICP Segment 1 (early-stage). Primary signal: seed extension, public
hires. Confidence: Medium-High. The PDF is a real artifact in the seed
repo, not invented.
GOOD #7 — Warm reply with bench routing to human
Subject Re: scope of the three-engineer engagement
## Body Hi Camila,
Thanks for the reply and for the additional context on the December
timeline.
Three Python and one data engineer for a 6-month engagement is in
our typical range. Our public-tier pricing for that mix starts from
approximately $X,XXX/month per engineer at senior level, with a
one-month minimum and two-week extension blocks thereafter.
A specific quote depends on the exact stack, the timezone overlap
requirement, and whether you want a Tenacious delivery lead
embedded. The cleanest path is a 30-minute scoping call with our
delivery lead, Arun. Here is his calendar: gettenacious.com/arun.
If the December start date is firm, I would suggest booking this week so
we can confirm capacity availability.
## Best,
## Yabi
Why it works
- Acknowledges the prior message context (December timeline) —
shows the thread is being read, not auto-replied.

- Quotes only the public band and explicitly routes a specific quote
to the human delivery lead.
- Names the dependencies that drive the quote, so the prospect
understands why a number is not on the email.
- Recommends booking this week with a real reason (capacity
confirmation), not fake urgency.
- Body 130 words — within the warm-reply 200-word limit.
Signal grounding Warm reply, ICP Segment 1. Bench match (3 Python + 1 data, senior,
6-month): supported by bench_summary.json. Pricing: only public
bands quoted; specific TCV correctly routed to human.
GOOD #8 — Re-engagement with new content, no guilt
Subject New: layoffs.fyi data on your sub-sector this quarter
## Body Hi Marcus,
When we last spoke in August, you mentioned that the board had not
yet pushed for cost rebalancing. Two new data points that may matter:
First, the layoffs.fyi data shows your sub-sector (vertical SaaS for
healthcare) had eleven announced contractions in the last 90 days, up
from four in the prior quarter. Boards are reading the same data.
Second, three of those eleven companies are now using
offshore-managed engineering teams within 60 days of restructure —
that pattern is faster than it was a year ago.
If the conversation has reopened on your side, our managed
engineering pricing has not changed. If not, no follow-up needed.
## Best,
## Yabi
Why it works
- New content carries the message — not "following up." The body
adds two specific, verifiable data points.
- References the prior conversation accurately ("in August," "the
board").
- Honest sub-sector framing — does not pretend the data points
are universal.
- Explicit "no follow-up needed" if the situation has not changed.
- Body 113 words. Re-engagement under the 100-word soft target
— slightly long, but the data justifies it.

Signal grounding Re-engagement, ICP Segment 2. Primary signal: layoffs.fyi sub-sector
data, fresh. Secondary: market behavior pattern (three of eleven
adopting offshore in 60 days).
GOOD #9 — Bench-gated honest decline of overreach
Subject Re: scaling to 15 engineers in 30 days
## Body Hi Will,
Thanks for the follow-up and for the trust to ask about the 15-engineer
ramp. Honest answer: 15 engineers across a Go and infra-heavy stack
within 30 days is at the edge of what our current capacity can deliver
responsibly.
What we can confirm now: 6 to 8 engineers in that stack, starting within
21 days, with a Tenacious delivery lead embedded. Going to 15 reliably
requires a 60-day ramp window, with the back half of the team
onboarding in weeks 5 and 6.
If the 30-day target is firm, I would rather refer you to a peer firm that
fits the timeline than over-commit. Happy to introduce.
## Best,
## Yabi
Why it works
- Refuses to over-commit — the most important capability of a B2B
services agent in production.
- Explains exactly what the bench can and cannot support, with a
clear timeline.
- Offers a referral to a peer firm if Tenacious cannot deliver —
earns long-term trust at the cost of a short-term deal.
- Body 116 words. Subject is direct and matches the substance.
Signal grounding Warm reply, hot stage. The 15-engineer Go/infra ask is not supported
by bench_summary.json. The agent correctly refused to commit and
offered a partial path plus a referral.
GOOD #10 — AI maturity 0–1 prospect, gentle Segment 1 reframe
Subject Question: standing up your first AI function
## Body Hi Sophia,

You closed your $9M Series A in March, your team is ten engineers, and
your public roles are all backend and product. No AI or ML postings yet
— which is a normal place to be at your stage, not a gap.
If your roadmap has an AI feature in the next twelve months, the first
hire is usually the wrong unit. A small dedicated squad (ML engineer
plus data platform engineer plus a Tenacious delivery lead) for a
3-month scoped project is faster, cheaper, and lets you test whether AI
is core enough to your roadmap to justify a full-time function.
If that is on your roadmap, 15 minutes to walk through what the first 90
days look like. If not, ignore this.
## Best,
## Yabi
Why it works
- Honest about the AI-maturity reading: "a normal place to be at
your stage, not a gap."
- Frames the pitch as the right first step — squad over hire —
rather than implying the prospect should already have AI
capacity.
- Conditional language throughout. Does not assume the AI
roadmap exists.
- Body 130 words. Subject matches the soft-pitch posture.
Signal grounding ICP Segment 1 with AI maturity 0–1. Funding signal: High confidence. AI
maturity gating: low score correctly triggers the Segment-1 "stand up
first AI function" framing rather than a Segment-4 capability-gap pitch.
GOOD #11 — Mutual connection (real), not name-drop
Subject Context: Arjun's recommendation
## Body Hi Mei,
Arjun Krishnan suggested I reach out — he and I worked on the data
platform redesign at his Series B in February, and he said your team is at
a similar stage with the same Snowflake plus dbt plus Airflow
combination he was working through.
If the equivalent rebuild is on your roadmap, I would be glad to share
what we learned in his project, including the two architectural decisions
that did not work and that Arjun would tell you about openly. Happy to
send a one-page write-up or do 15 minutes — your call.

If this is not on your roadmap, no follow-up.
## Best,
## Yabi
Why it works
- Names the mutual connection and the specific project context —
Arjun would recognize this if asked. No fabrication.
- References specific stack components (Snowflake, dbt, Airflow)
that match the prospect's actual stack — would be from the
BuiltWith signal in the brief.
- Names two decisions that did not work — earns trust through
specificity.
- Two options offered (PDF or call), low friction.
- Body 110 words.
Signal grounding ICP Segment 4 (capability gap, data platform). Primary signal: warm
intro plus matching tech stack. Confidence: High. The Arjun reference
would need to be a real prior client of Tenacious.
GOOD #12 — Two-line micro-touch, post-engagement nurture
Subject Quick thought after our call
## Body Hi Kevin,
After we spoke yesterday I went back and looked — three of the loyalty
platforms you mentioned as competitors are now publicly using the
same dbt-plus-Snowflake stack you are evaluating. Worth knowing as
you scope the build.
No reply needed. I will follow up after your internal review next
Thursday as agreed.
## Best,
## Yabi
Why it works
- Two-line micro-touch — adds one specific, useful data point, then
steps back.
- "No reply needed" respects the prospect's time.
- References the prior call accurately, including the agreed
next-step date.
- Body 47 words. Subject is short and honest.

Signal grounding Warm-stage post-call nurture, ICP Segment 4. Primary signal:
competitor stack data from BuiltWith. Confidence: Medium. The
follow-up date matches the discovery-call brief; do not hallucinate
dates.
Twelve Bad Drafts (Labeled, with Failure Modes)
Each bad draft fails one or more tone markers, formatting constraints, or pre-flight checks. The failure
modes are tagged so the agent's tone-preservation check can learn the patterns.

BAD #1 — Wall of self-promotion (Me, Me, Me)
Subject Tenacious — World-Class Engineering Talent
## Body Dear Maya,
Tenacious Intelligence Corporation is a world-class engineering
outsourcing firm with over 200 senior engineers across Python, Go,
data, ML, and infrastructure. We have been operating since 2019 and
have served clients across North America and Europe. Our top talent is
graduated from elite programs and our delivery model is the gold
standard in the industry.
We offer junior, mid, senior, and architect-level engineers, fractional
CTO services, project consulting on AI systems, data platforms, and
specialized infrastructure, training engagements, and strategic AI
workshops. Our pricing is highly competitive and we offer flexible
engagement models.
I would love to schedule a 45-minute discovery call to learn about your
business, your goals, your pain points, your budget, and your roadmap
so we can determine the best way to partner.
Best regards,
## Yabi
Why it fails
- "World-class," "top talent," "gold standard" — three banned
phrases in the first paragraph.
- Self-centered: 80% of the body is about Tenacious, not the
prospect.
- No hiring signal grounded — Maya could be any prospect; the
email would read identically to any other recipient.

- Lists every service offered (Me, Me, Me #1 from generic
anti-pattern set).
- Asks for 45 minutes to learn the prospect's budget, pain points,
and roadmap in one call (Unpaid Homework).
- Body 152 words — exceeds the 120-word limit.
- Subject contains a banned phrase.
Signal grounding No signals from the brief used. This draft would be sent identically to
every prospect — the failure mode that prompted Tiffany Gonzalez's
"weirdest cold outreach" post.
BAD #2 — Asserts on weak signal
Subject Quick chat: your aggressive hiring
## Body Hi Tom,
I see you are scaling aggressively — your engineering team is clearly
growing fast and you must be feeling the pain of recruiting velocity right
now. Companies in your stage always hit a wall around month four after
a Series A.
We solve this exact problem. Tenacious places top talent in 48 hours
and we will skyrocket your delivery throughput.
Quick question — do you have 15 minutes this week?
## Best,
## Yabi
Why it fails
- "Scaling aggressively" asserted on a weak signal (Tom has 2 open
roles in the brief, not 7).
- "Always hit a wall around month four" — a generalization
presented as a specific insight about Tom's company.
- "Top talent" and "skyrocket" — two banned phrases.
- Subject starts with "Quick chat" — banned formulation.
- Assumes a pain ("you must be feeling") that the signal does not
support.
- Body 72 words but loses on every honesty marker.
Signal grounding Signal in brief: 2 open roles, signal_status "Weak signal — ask, do not
assert." The agent ignored the confidence flag and asserted anyway.
This is the most common Honest-marker failure mode.

BAD #3 — Bench overcommitment
Subject Re: 12 senior Go engineers in two weeks
## Body Hi Will,
Absolutely, we can deliver 12 senior Go engineers in two weeks. Our
bench is deep across all stacks and we move fast.
I will have our team kick off scoping immediately. Expect contracts by
Wednesday and engineers in your Slack by next Friday.
## Best,
## Yabi
Why it fails
- Commits 12 senior Go engineers in two weeks — not supported
by bench_summary.json.
- Uses the word "bench" externally — Professional-marker failure.
- Implies contracts and engineer onboarding without scope,
without pricing, without delivery-lead handoff. This is a
Tenacious-policy violation.
- Skipped the bench-to-brief check entirely.
- Body is short, but every line is a violation.
Signal grounding Bench reality: Go has 4 senior engineers available, not 12. The agent
committed unsupported capacity, which is a disqualifying violation in
the Week 11 grading rubric. Correct behavior: route to human (see
## GOOD #9).
BAD #4 — Condescending competitor gap framing
Subject Your AI maturity is behind the curve
## Body Hi Felix,
I will be direct: your AI maturity score is a 1, while your top competitors
are a 3. You are falling behind in a market where AI is no longer
optional, and your leadership has not yet made the strategic moves that
the loyalty-platform sector demands in 2026.
Tenacious can stand up your missing MLOps function and close the gap
before your next board meeting. Our agentic systems and ML platform
engineers are world-class.
Let's get on a call this week to discuss how we can help you catch up.
## Best,

## Yabi
Why it fails
- Subject itself is condescending — "behind the curve." Senior
engineering leaders will not respond to this.
- "You are falling behind," "has not yet made the strategic moves,"
"close the gap," "catch up" — the entire frame is the prospect
failing.
- "World-class" — banned phrase.
- "AI is no longer optional" — empty market claim.
- Assumes a board meeting and AI strategy gap that are not in the
brief.
- Body 102 words but every paragraph violates the
Non-condescending marker.
Signal grounding Same signal data as GOOD #4 (three peers with public MLOps openings,
prospect has none, AI maturity score 2). The agent correctly identified
the gap but framed it as the prospect's failure rather than a research
finding. Tone failure on Non-condescending and Honest.
BAD #5 — Aggressive third follow-up
Subject Per my last three emails
## Body Hi Ana,
I have sent you three emails over the last two weeks and have not
heard back. I have to assume you are not interested in growing your
engineering capacity, which is fine — but I would appreciate a one-line
reply to confirm so I can take you off the list.
If I do not hear back by Friday, I will assume the answer is no.
## Best,
## Yabi
Why it fails
- "Per my last three emails," "I have to assume you are not
interested" — passive-aggressive, banned formulation.
- Assumes intent on the prospect's behalf — Honest-marker failure
and Non-condescending failure simultaneously.
- Implies guilt for not replying — exactly the re-engagement
anti-pattern.
- Fake deadline ("by Friday") with no real consequence.
- Subject is a banned phrase.

Signal grounding Re-engagement context. Correct behavior is GOOD #8: re-engage with
new content, not guilt. The agent failed to use the layoffs.fyi or other
fresh signal that was available in the brief.
BAD #6 — Generic templated outreach
Subject Hey [First Name], scaling your engineering team?
Body Hey [First Name],
I hope this email finds you well. I am reaching out because I think
Tenacious can help [Company] with all of your engineering and AI needs
in 2026.
We work with companies like yours to deliver world-class talent at
affordable prices. Our team has experience across many industries and
stacks, and we can help you scale, restructure, or build new capabilities
depending on what you need.
Would you be open to a quick chat next week to explore how we can
synergize and add value to your ecosystem?
## Best,
## Yabi
Why it fails
- Unfilled template tokens — the agent failed even to populate
[First Name] and [Company].
- "I hope this email finds you well" — banned opener.
- "World-class," "synergize," "add value," "ecosystem," "quick chat"
— five banned phrases.
- "Many industries and stacks," "depending on what you need" —
vague and self-centered.
- No signal grounded. This email could be sent to any prospect on
## Earth.
- Body 92 words but fails every tone marker.
Signal grounding This is the Tiffany Gonzalez "weirdest cold outreach" archetype —
high-volume, zero-personalization, every recipient gets the same body.
Tone failure on every marker.
BAD #7 — Fake urgency / discount
Subject URGENT: Last open slot for Q1 — 30% off if you sign this week
## Body Hi Kevin,

Tenacious has one remaining slot in our Q1 cohort for managed
engineering teams. Because of strong demand, this slot will not be
available after Friday.
If you sign a contract by end of day Friday, I am authorized to offer 30%
off the first three months. After that, the slot goes to the next company
on the waitlist.
Do not miss out on this — book a call here: gettenacious.com/yabi.
## Best,
## Yabi
Why it fails
- Fake scarcity — "one remaining slot," "strong demand," "waitlist"
— none of which is grounded in reality.
- Discount that is not in pricing_sheet.md — the agent invented a
30% off promotion.
- "Do not miss out" — banned formulation.
- "Authorized to offer" implies a sales-rep escalation that the agent
does not have.
- Subject in all caps with "URGENT" — every recipient screens this
as spam.
Signal grounding Tenacious does not run discount campaigns and does not have a Q1
cohort waitlist. Inventing pricing structures is a disqualifying violation of
the pricing-scope check.
BAD #8 — Wrong segment pitch (capability gap on AI maturity 0)
Subject Question: your agentic systems roadmap
## Body Hi Sophia,
I am curious how you are thinking about your agentic-systems roadmap
for 2026. Most peer companies in your stage are now scoping
LLM-orchestrated workflows and dedicated MLOps functions to support
production agent deployments.
We staff specialized capability-gap squads — agentic systems, ML
platform, data contracts — typically 3 to 4 months. Starter scope from
$XX,XXX. We have done this for several Series A and B SaaS companies
in the last year.
Want to set up a 30-minute scoping conversation?
## Best,

## Yabi
Why it fails
- Pitches a Segment 4 capability-gap engagement to a prospect
with AI maturity score 0–1.
- Assumes an agentic-systems roadmap that the brief does not
support.
- "Most peer companies in your stage are now scoping..." —
generalization presented as fact, no specific peers named.
- Wrong-audience: Sophia's team is ten people, all backend and
product, no AI roles. The agent should have used the GOOD #10
framing.
- Body 96 words but the segment selection is the failure.
Signal grounding Correct behavior at AI maturity 0–1 is the Segment 1 reframe ("stand
up your first AI function"), not the Segment 4 capability-gap pitch. The
AI maturity gating rule was ignored.
BAD #9 — PDF attachment cold (no consent)
Subject Tenacious capabilities deck — review pages 8 and 12
## Body Hi Marcus,
Please find attached our 38-page capabilities deck.
Pages 8 and 12 are the most relevant to your sub-sector. Let me know
your thoughts and we can schedule a call to discuss our partnership
opportunity.
Looking forward to your reply.
## Best,
## Yabi
[ATTACHMENT: tenacious_capabilities_v7.pdf — 12.4 MB]
Why it fails
- Cold PDF attachment — instant spam-filter trigger and the lowest
reply rate of any cold-outreach format.
- No signal grounded. Marcus is asked to do the work of finding
what is relevant to him on pages 8 and 12.
- "Partnership opportunity" — vague consultant language.
- Body is empty of substance — the agent has outsourced the value
to the PDF.
- Cold attachments violate the formatting constraint explicitly.

Signal grounding Cold PDF attachments are banned in the formatting rules. Send a link to
a one-page web page instead, after the prospect has expressed interest.
See GOOD #6 for the right pattern.
BAD #10 — Multiple stacked asks
Subject A few questions and ideas for [Company]
## Body Hi Daniel,
I had a few thoughts I wanted to share. First, I would love to understand
your current engineering structure and which stacks you are using.
Second, I have an introduction to a peer of yours at a similar
mid-market platform that I think you should meet. Third, we have a
new training program for engineering leaders that might be relevant.
Fourth, I noticed your AI maturity is around a 2 — happy to walk
through how to move it to a 3.
Could we set up a 60-minute call next week to discuss all four of these?
I will also send our pricing sheet, our case studies, and our training
brochure separately.
## Best,
## Yabi
Why it fails
- Four asks in one message — violates the one-ask rule.
- 60-minute call — too long for a first conversation.
- "Send our pricing, case studies, and brochure separately" — three
additional emails the prospect did not consent to.
- "Move your AI maturity from 2 to 3" — frames AI maturity score
as Tenacious's problem to solve, condescending.
- Body 121 words but unfocused. The prospect cannot decide what
to reply to first.
Signal grounding Pick the highest-value ask and ship one. The four-asks anti-pattern is
the single most common reason a B2B prospect leaves a thread on
read.
BAD #11 — Pricing fabrication for multi-phase engagement
Subject Quote: $1.2M for the 12-month engagement
## Body Hi Camila,

Thanks for the call yesterday. As discussed, our quote for the 12-month
engagement covering 6 engineers, a delivery lead, and a fractional
architect is $1,200,000 total, payable in monthly installments of
## $100,000.
I have attached the contract. Please sign and return by Friday so we can
begin onboarding on the 1st.
## Best,
## Yabi
Why it fails
- Total contract value invented — pricing_sheet.md explicitly
forbids the agent from quoting multi-phase TCVs.
- Contract attached — far outside the agent's scope; only humans
send contracts.
- Hard signing deadline without negotiation room.
- Skipped the route-to-human policy that the pricing scope check
requires.
- Body 67 words. The brevity makes the violation worse — the
prospect feels rushed into a million-dollar commitment.
Signal grounding Multi-phase pricing requires human delivery-lead handoff. The agent's
pricing-scope check should have flagged this draft and routed it. See
GOOD #7 for the correct routing pattern.
BAD #12 — Signal fabrication
Subject Re: your $40M Series C
## Body Hi Priya,
Congratulations on closing your $40M Series C last month — exciting
moment for the team. With that level of capital, scaling engineering
aggressively is the obvious next move.
We can plug a 15-engineer team into your stack within 30 days at our
standard rates. Junior engineers from $X,XXX/month.
Want to set up a 15-minute call to discuss?
## Best,
## Yabi
Why it fails
- Series C and $40M figure are fabricated — the brief shows Priya's
company at Series A, $9M, 14 months ago.

- Confident assertion of a buying window that does not exist.
- "Scaling engineering aggressively is the obvious next move" —
assumes a strategy the prospect has not announced.
- Bench commitment of 15 engineers in 30 days — not supported.
- If Priya forwards this email, every claim is verifiably wrong. Brand
damage from one bad email outweighs a week of good ones.
Signal grounding This is the highest-cost failure mode in the rubric. Hallucinated funding
events, leadership changes, layoffs, or AI maturity scores are
disqualifying violations because the prospect can verify them in
seconds.
Tone-Preservation Check Specification
This specification defines how the agent (and the Path B trained judge, if you choose that path) scores
any draft against the five tone markers. Each marker is scored 1–5; a draft scoring below 4/5 on any
marker is regenerated.
## Marker Score 5 (passes)
## Score ≤ 2 (fails)
## Direct
Subject states intent, body ≤ 120 words for
cold (≤ 200 warm), one ask, no filler.
"Quick / Just / Hey" subject;
multi-paragraph self-introduction; ≥
2 asks.
Grounded At least one specific signal from the brief
named (amount, date, role count, named
peer). Confidence-aware phrasing matches
signal confidence in the brief.
No signal named, or signal asserted
at higher confidence than the brief
supports ("aggressive hiring" on 2
open roles).
Honest Names what the brief does not show. Uses
interrogative phrasing for low-confidence
signals. Refuses to commit bench or pricing
the policy does not allow.
Asserts unsupported claims; commits
bench beyond bench_summary.json;
quotes a TCV outside the public
bands.
Professional No banned phrases. No "bench" externally.
Language calibrated to a CTO or founder
reader.
Banned phrase present; "bench"
used to a prospect; consultant jargon
## ("synergize," "ecosystem").
## Non-condesc
ending
Frames any gap as a research finding or
question, with explicit acknowledgement
"Falling behind," "missing," "behind
the curve," "you need to," "you

that the prospect may have already
considered it.
should"; assumes prospect's
reasoning.
The LinkedIn-Roast Test
Before any draft sends, ask: if this email were screenshotted and posted on LinkedIn with the prospect's
annotation, would it be roasted?
Cold outreach posts that go viral on LinkedIn have a recurring pattern. They land in the inbox of a senior
leader during a quiet moment (a holiday, a board prep day, a flight). The leader screenshots them with a
one-line caption like "this is what cold outreach looks like in 2026" or "please tell me this is
AI-generated." The post collects a few thousand reactions. The sending firm's name is now associated
with the failure mode in front of the leader's network — exactly the buyer cohort the firm wanted to
reach.
The reputational asymmetry is severe: 1,000 emails with one viral failure costs more in trust than 1,000
emails with no failures gain. Any draft that could plausibly become that screenshot is regenerated.
Failure modes that get screenshotted most often:
- The unsolicited "COO offer" — pitching a senior strategic role to a founder who already has one.
- The fabricated funding event — congratulating on a round that did not happen, or wrong by a
stage.
- The condescending gap analysis — "your AI maturity is behind your competitors."
- The aggressive third follow-up — "per my last three emails."
- The unfilled template token — "Hi [First Name]."
## Outreach Decision Flow
Before the agent composes any draft, it runs through this flow. The flow determines the segment
framing, the language register, and the routing rules.
## Step Question Action
1 Which ICP segment does the
prospect fit, and at what
confidence?
Read primary_segment_match and signal confidences
from hiring_signal_brief.json. If confidence is Low across
all signals, default to a value-add resource touch (GOOD
#6 pattern), not a pitch.

2 What is the AI maturity
score, and at what
confidence?
Score 0–1: Segment 1 "stand up your first AI function"
framing or no AI mention. Score 2: Segment 1 or 2 with AI
optionality. Score 2 with high confidence and named peer
gaps: Segment 4 capability-gap framing (GOOD #4). Never
pitch Segment 4 below score 2.
3 Does the brief commit to
specific capacity, stack, or
timeline?
Cross-check against bench_summary.json. If the bench
supports it, commit specifically (GOOD #1). If not, route to
human (GOOD #9).
4 Does the draft mention
price?
Quote only public bands from pricing_sheet.md (junior
monthly floor, senior monthly, starter project floor, hourly
blended). Multi-phase TCV always routes to a human.
5 Has the prospect been
contacted before?
If yes and the previous thread stalled, re-engagement only
with new content (GOOD #8). Never "following up."
6 Final pre-flight scan Run banned-phrase check, word-count check, signature
check, and the LinkedIn-roast test. Fail any → regenerate.

Ground every claim. Refuse what cannot be supported. Send what you would defend.