# AI Draft Bot - Use Case Specification

**Version:** 1.0
**Date:** 2025-11-23
**Status:** Active

---

## 1. Product Overview

### What is AI Draft Bot?

**AI Draft Bot** is a machine learning-powered assistant that provides real-time pick recommendations during Magic: The Gathering Limited format drafts (Booster Draft and Premier Draft). The system analyzes the current pack, your deck-in-progress, and historical win rate data to recommend the optimal pick with human-expert level accuracy (75-85%).

### Value Proposition

**For casual players:**
- Improve draft performance without years of experience
- Learn which cards are actually good (not just flashy)
- Build more cohesive decks with better mana curves

**For competitive players:**
- Validate your instincts with data-driven insights
- Discover hidden gems in unfamiliar formats
- Optimize pick decisions in complex situations

**For content creators:**
- Explain draft decisions with SHAP explanations
- Compare your picks against AI recommendations
- Generate content around "human vs AI" drafting

---

## 2. Primary Use Cases

### Use Case 1: Real-Time Draft Assistance (Arena/MTGO)

**Actor:** Player drafting on Magic: Arena or MTGO
**Trigger:** Pack opens with 13-15 cards
**Flow:**
1. Browser extension captures pack contents via screen scraping
2. Extension sends pack + current deck to API endpoint
3. API returns ranked recommendations in <100ms
4. Extension displays recommendations as overlay
5. Player makes final decision (with or without AI guidance)

**Success Criteria:**
- Latency <100ms (real-time)
- Accuracy 75-85% (matches expert picks)
- Non-intrusive UI (optional overlay)

**Example:**
```
Pack: [Lightning Bolt, Counterspell, Llanowar Elves, ...]
Deck: [Swords to Plowshares, Path to Exile, ...]

AI Recommendation:
1. Lightning Bolt (94% confidence)
   → High win rate, fits your white-red aggro deck
2. Counterspell (78% confidence)
   → Powerful card, but off-color
3. Llanowar Elves (52% confidence)
   → Ramp, but deck needs removal
```

---

### Use Case 2: Post-Draft Analysis

**Actor:** Player reviewing completed draft
**Trigger:** Draft finished, wants to learn from mistakes
**Flow:**
1. Player uploads draft log (17Lands export or manual entry)
2. System re-simulates draft with AI recommendations
3. System highlights disagreements (picks where player ≠ AI)
4. System explains why AI recommended different card
5. Player learns from analysis

**Success Criteria:**
- Batch processing of full draft (42 picks)
- Clear explanations for each disagreement
- Exportable report (PDF/JSON)

**Example:**
```
Pack 1, Pick 3: You picked "Grizzly Bears"
AI recommended: "Lightning Bolt" (89% confidence)

Reason: Lightning Bolt has 62% GIH win rate vs Bears at 51%.
        Removal is more valuable in this format.
        Your deck has 0 removal so far.
```

---

### Use Case 3: Format Learning Tool

**Actor:** New player learning unfamiliar format
**Trigger:** New set released, player wants to learn quickly
**Flow:**
1. Player practices drafts with AI guidance enabled
2. AI explains *why* each card is good/bad
3. Player learns format-specific priorities (e.g., "removal is premium in this set")
4. Over time, player internalizes these lessons

**Success Criteria:**
- Natural language explanations (not just numbers)
- Archetype identification ("You're building WU Fliers")
- Meta insights ("Red is overdrafted this week")

**Example:**
```
Why is "Lightning Bolt" better than "Giant Growth"?

✓ Removal vs pump spell: Removal is more flexible
✓ Win rate: Bolt 62% vs Growth 49%
✓ Deck synergy: You're controlling, not aggressive
✓ Format context: Bombs are common, need answers
```

---

## 3. Secondary Use Cases

### Use Case 4: Sealed Deck Building (Future)

**Actor:** Player building 40-card deck from 6 boosters
**Trigger:** Sealed event starts
**Flow:**
1. Player enters 90 cards from sealed pool
2. AI suggests 2-3 optimal deck configurations
3. Player reviews and tweaks suggested decks
4. AI explains why certain cards made the cut

**Status:** Not currently supported (requires different feature set)
**Priority:** Medium (smaller user base than draft)

---

### Use Case 5: Opponent Modeling (Partial)

**Actor:** Experienced player tracking opponent signals
**Trigger:** Card wheels (comes back in next pack)
**Flow:**
1. AI detects wheeled cards (unusual late picks)
2. AI infers opponents' color preferences
3. AI suggests "open" colors with less competition
4. Player adjusts strategy accordingly

**Status:** Partial (color competition features exist, not exposed in API)
**Priority:** Low (advanced use case)

---

### Use Case 6: Training Data for Meta-Models (Research)

**Actor:** Researcher/developer building higher-level systems
**Trigger:** Need card embeddings or win rate predictions
**Flow:**
1. Export card embeddings from LSTM model
2. Use embeddings for downstream tasks (deck building, matchup prediction)
3. Query API for card quality estimates

**Status:** Supported via API
**Priority:** Low (niche use case)

---

## 4. User Personas

### Persona 1: "Competitive Cam"
- **Background:** Plays 10+ drafts per week, aims for Mythic rank
- **Pain Points:** Unfamiliar formats, high-stakes decisions, tilt from bad picks
- **Goals:** Maximize win rate, learn new sets quickly, validate instincts
- **Usage:** Real-time overlay during Arena drafts, post-draft analysis
- **Willingness to Pay:** High ($10-20/month)

### Persona 2: "Casual Carla"
- **Background:** Plays 2-3 drafts per month, mostly for fun
- **Pain Points:** Doesn't know which cards are good, loses to better drafters
- **Goals:** Have more fun, win occasionally, learn gradually
- **Usage:** Real-time assistance, occasional explanations
- **Willingness to Pay:** Low ($0-5/month)

### Persona 3: "Content Creator Chris"
- **Background:** Streams drafts on Twitch/YouTube
- **Pain Points:** Needs unique content, wants to educate viewers
- **Goals:** "Human vs AI" content, explain draft theory, engage chat
- **Usage:** Real-time overlay (visible to stream), explanations for viewers
- **Willingness to Pay:** Medium ($5-15/month, tax-deductible)

### Persona 4: "Researcher Rachel"
- **Background:** Data scientist studying game AI
- **Pain Points:** Needs structured data and APIs
- **Goals:** Research publications, benchmark algorithms
- **Usage:** API access, embeddings export, batch processing
- **Willingness to Pay:** Variable (academic/commercial)

---

## 5. Integration Scenarios

### Scenario A: Browser Extension (Chrome/Firefox)
**Description:** Lightweight extension overlays recommendations on Arena/MTGO
**Technical Requirements:**
- Screen scraping to detect pack contents
- <100ms API latency for real-time UX
- Non-intrusive overlay (draggable, resizable)
- Offline mode with cached model (optional)

**Implementation Complexity:** Medium (3-4 weeks)

---

### Scenario B: Mobile App (iOS/Android)
**Description:** Companion app for paper drafts at LGS events
**Technical Requirements:**
- Manual card entry (camera OCR optional)
- Offline mode (on-device quantized model)
- Deck tracking across draft
- Export draft log for later analysis

**Implementation Complexity:** High (8-12 weeks)

---

### Scenario C: Web Dashboard
**Description:** Post-draft analysis and learning tool
**Technical Requirements:**
- Upload 17Lands draft log
- Interactive visualization of pick-by-pick decisions
- Archetype detection and meta insights
- Exportable reports

**Implementation Complexity:** Medium (4-6 weeks)

---

### Scenario D: Discord Bot
**Description:** Ask AI for draft advice in Discord
**Technical Requirements:**
- `/draft pack:[card1,card2,...] deck:[...]` command
- Responds with top 3 recommendations + explanations
- Remembers draft context per user session
- Rate limiting per server

**Implementation Complexity:** Low (1 week)

---

## 6. Non-Goals (Out of Scope)

### Explicitly Not Supported
1. **In-game play decisions:** No advice during gameplay (only drafting)
2. **Constructed deck building:** Only Limited formats (not Standard, Modern, etc.)
3. **Card price tracking:** Not a financial tool
4. **Tournament rule violations:** Complies with WotC policies
5. **Fully automated drafting:** Human must make final decision

### Why These Are Out of Scope
- **In-game play:** Different problem domain (game tree search, not ML)
- **Constructed:** Different data source (metagame ≠ Limited format)
- **Pricing:** Not our core competency
- **Rules:** Must comply with Magic's Tournament Rules (no unauthorized assistance during sanctioned events)
- **Automation:** Ethical concerns, violates Arena ToS

---

## 7. Success Metrics

### Product Metrics
- **Daily Active Users (DAU):** Target 500+ within 6 months
- **Retention Rate:** 40%+ weekly retention
- **API Requests:** 10,000+ predictions per day
- **Draft Completion Rate:** 80%+ of drafts completed with AI assistance

### Quality Metrics
- **Prediction Accuracy:** 75-85% (matches expert picks)
- **API Uptime:** >99.9%
- **Latency:** <100ms p95
- **User Satisfaction:** 4.5+ stars (if app store)

### Business Metrics (if monetized)
- **Conversion Rate:** 10%+ free → paid
- **ARPU:** $8-12/month
- **Churn Rate:** <5% monthly
- **NPS Score:** 50+ (promoters - detractors)

---

## 8. Competitive Analysis

### Existing Solutions

| Product | Strengths | Weaknesses | Our Advantage |
|---------|-----------|------------|---------------|
| **17Lands** | Comprehensive data, trusted source | No AI recommendations, manual analysis | Real-time AI, explanations |
| **DraftSim** | Free, web-based simulator | Static ratings, not ML-based | Dynamic context-aware picks |
| **Untapped.gg** | Arena overlay, deck tracking | No draft assistance (yet) | Draft-specific AI |
| **Human Experts** | Intuition, meta knowledge | Expensive ($200+ coaching), not scalable | Accessible, instant |

### Differentiation Strategy
1. **Accuracy:** Match human experts (75-85%) with ML
2. **Speed:** <100ms real-time recommendations
3. **Explanations:** SHAP-based insights (not black box)
4. **Always Updated:** Automated retraining with latest data
5. **Accessible:** Free tier + affordable premium

---

## 9. Regulatory and Ethical Considerations

### WotC Tournament Rules
**Question:** Is AI assistance legal during drafts?

**Answer:**
- ✅ **Legal:** Casual play, practice drafts, post-draft analysis
- ⚠️ **Gray Area:** Arena ranked drafts (not officially sanctioned)
- ❌ **Illegal:** Sanctioned paper tournaments (REL Competitive+)

**Our Stance:**
- Clearly label tool as "practice and learning aid"
- Discourage use in sanctioned tournaments
- Promote ethical use (human makes final decision)

### Data Privacy
- No personal data collected (only pack contents)
- No tracking of individual players
- No sharing of draft logs without consent
- GDPR-compliant if EU users

### Fair Play
- Tool augments human decision-making (not replaces)
- Explanations promote learning (not blind following)
- Offline mode prevents dependency on internet

---

## 10. Future Enhancements (Backlog)

### Short-Term (3-6 Months)
1. **Archetype Recommendation:** "You're building WU Fliers, prioritize evasion"
2. **Meta Tracking:** "Red is 15% overdrafted this week"
3. **Deck Evaluation:** "Your deck is a B+ with good curve, weak removal"

### Medium-Term (6-12 Months)
1. **Sealed Support:** Build optimal deck from sealed pool
2. **Opponent Modeling:** "Left player is likely in black, pivot to green"
3. **Card Price Integration:** "This $20 card is also a strong pick"

### Long-Term (12+ Months)
1. **Reinforcement Learning:** AI learns from wins/losses, not just picks
2. **Multi-Format Support:** Cube drafts, chaos drafts, custom formats
3. **Social Features:** Share drafts, compare with friends, leaderboards

---

## 11. Open Questions

### Product Questions
1. **Monetization:** Free + premium tiers, or fully open-source?
2. **Branding:** "AI Draft Bot" or catchier name?
3. **Platform Priority:** Arena extension first, or mobile app?

### Technical Questions
1. **Model Selection:** XGBoost (fast) or LSTM (accurate) as default?
2. **Hosting:** Self-hosted or cloud (AWS/GCP)?
3. **Caching:** Redis for request caching, or in-memory only?

### Legal Questions
1. **WotC Permission:** Should we seek official blessing?
2. **17Lands Data:** License for using their data exports?
3. **Arena ToS:** Does screen scraping violate Terms of Service?

---

## 12. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-23 | Primary use case: Real-time draft assistance | Highest user value, clear differentiation |
| 2025-11-23 | Target accuracy: 75-85% | Matches human experts, achievable with current models |
| 2025-11-23 | API-first architecture | Enables multiple clients (web, mobile, Discord) |
| 2025-11-23 | Free tier supported | Lower barrier to entry, build user base |

---

## Appendix: User Flows

### Flow 1: First-Time User (Arena Extension)

```
1. User installs Chrome extension
2. Extension shows onboarding: "AI Draft Bot will suggest picks during drafts"
3. User starts Arena draft
4. Pack opens → Extension detects cards
5. Extension shows overlay: "Lightning Bolt (94%) - High win rate, removal"
6. User clicks "Why?" → SHAP explanation appears
7. User picks Lightning Bolt (or ignores AI)
8. Repeat for 42 picks
9. Draft ends → Extension shows summary: "You agreed with AI 28/42 times (67%)"
10. Extension prompts: "Upgrade to premium for post-draft analysis"
```

### Flow 2: Experienced User (API Access)

```
1. Developer reads API_README.md
2. Developer gets API key (if required)
3. Developer sends POST /predict with pack + deck
4. API returns JSON with ranked recommendations
5. Developer integrates into custom tool (Discord bot, Twitch overlay, etc.)
6. Developer monitors /health endpoint
7. Developer reviews /metrics for usage analytics
```

---

**Document Owner:** AI Draft Bot Team
**Last Updated:** 2025-11-23
**Next Review:** 2025-12-23
