# LegoGen: Build Guidance + Explore Page Design Spec

**Date:** 2026-04-04
**Branch:** training-v3
**Goal:** Transform LegoGen from "just a model" into a full platform with real-time build guidance, model comparison, and a build gallery — demonstrating system engineering, ML evaluation, and creative interaction design.

---

## 1. Build Guidance Mode (Hero Feature)

### 1.1 Overview

A dedicated `/guide/:buildId` page with split-screen layout. Left side shows the user's webcam feed (physical workspace). Right side shows an interactive 3D reference viewer locked to the current build step, with visual highlights guiding brick placement.

### 1.2 Layout

```
+-----------------------------------------------------+
|  [<- Back]     BUILD GUIDANCE MODE        [Step 3/8] |
+----------------------+------------------------------+
|                      |                              |
|    WEBCAM FEED       |    3D REFERENCE VIEW         |
|   (your workspace)   |  (current step highlighted,  |
|                      |   next brick pulsing green)  |
|                      |                              |
+----------------------+------------------------------+
| < Prev |  >> Play  | Next > |  "Place the 2x4 red  |
|        |           |        |   brick on top of     |
|        |           |        |   the base plate"     |
+-----------------------------------------------------+
| PARTS CHECKLIST                                     |
| [x] 1x Baseplate 32x32 (Green)                     |
| [ ] 2x Brick 2x4 (Red)          <- current         |
| [ ] 1x Plate 1x2 (White)                           |
+-----------------------------------------------------+
```

### 1.3 3D Viewer Enhancements (GuidanceViewer)

- **Ghost bricks:** All future steps rendered as wireframe/transparent outlines (opacity 0.15). User sees the final shape while building.
- **Current step bricks:** Full color, fully opaque, subtle drop shadow.
- **Pulsing highlight:** The brick currently being narrated gets a green emissive glow animation cycling via `useFrame`.
- **Camera auto-framing:** On step advance, camera smoothly lerps to focus on the area where new bricks are placed.
- **Exploded view toggle:** Button that spreads steps apart vertically to show layer structure.
- **Previous steps:** Rendered at 50% opacity so context is visible but current step stands out.

### 1.4 Shared Brick Component

Extract common brick rendering from existing `LegoViewer.tsx` into a shared `BrickMesh.tsx` component:
- Renders a brick given: part type, position, color, opacity, wireframe flag, glow flag
- Stud rendering on top
- Configurable geometry based on part dimensions
- Used by both `LegoViewer.tsx` (existing BuildSession view) and `GuidanceViewer.tsx`

### 1.5 Voice Narration

- Browser-native `Web Speech API` (`window.speechSynthesis`) — zero external dependencies.
- When a step advances, the step instruction string is spoken aloud.
- Controls: mute/unmute toggle, speed selector (0.75x / 1x / 1.25x).
- Implementation: `VoiceNarrator.ts` utility class wrapping SpeechSynthesis.

### 1.6 Auto-Play Mode

- Play button walks through steps automatically.
- Sequence per step: speak narration -> wait for speech to finish -> configurable pause (default 10s) -> advance.
- Progress bar shows time remaining until next step.
- Pause/resume at any time.

### 1.7 Step Timer

- Per-step timer starts when entering a step.
- Total elapsed build time in the header bar.
- Build completion summary: "Build complete! Total time: 4m 32s".

### 1.8 Parts Checklist

- Interactive checklist for the current step's parts.
- User can check off parts as they place them.
- Visual indicator for current part (highlighted row).
- Shows part name, color swatch, quantity.

### 1.9 Entry Point

From BuildSession page, after a build is generated, a **"Start Building"** button navigates to `/guide/:buildId`. Also accessible from the gallery ("Build This" button).

**Data flow for guidance mode:**
- **From BuildSession:** Build data (description JSON + steps) is passed via React Router state (`navigate('/guide/new', { state: { build } })`). No backend call needed.
- **From Gallery:** Build data is fetched via `GET /api/gallery/:id` using the buildId URL param, then steps are computed client-side from the description JSON using the same logic as BuildSession.

---

## 2. Explore Page (Comparison + Gallery)

### 2.1 Overview

A single `/explore` page with two tabs: **Compare** and **Gallery**. Combines model evaluation (showing ML rigor) with build persistence (showing software engineering).

### 2.2 Tab 1: Model Comparison

**Purpose:** Show side-by-side outputs from different training checkpoints, proving the model improved through iteration.

**Data source:** Pre-computed results for ~20 test inputs across available checkpoints:
- Vision: checkpoints 45, 90, 135
- Planner: checkpoints 200, 400, 600

**Pre-computation:** A script `scripts/precompute_comparisons.py` runs inference on each checkpoint for each test input, stores results as static JSON in `data/comparisons/`.

**UI layout:**
- Dropdown to select test input (e.g., "Red Sports Car", "Medieval Castle")
- Side-by-side panels showing each checkpoint's output:
  - 3D preview (mini LegoViewer)
  - Metrics: JSON validity, parts F1, color F1, part count, complexity
- Summary bar chart across all 20 test inputs showing aggregate metrics per checkpoint
- Bar chart rendered with pure CSS (no charting library)

### 2.3 Tab 2: Build Gallery

**Purpose:** Users can save generated builds, browse them, and launch guidance mode from any saved build.

**Backend storage:** SQLite via `aiosqlite`.

**Schema:**
```sql
CREATE TABLE builds (
    id TEXT PRIMARY KEY,        -- UUID
    title TEXT NOT NULL,
    category TEXT,
    complexity TEXT,
    parts_count INTEGER,
    description_json TEXT,      -- full LegoDescription JSON
    thumbnail_b64 TEXT,         -- base64 PNG from canvas capture
    stars REAL DEFAULT 0,
    star_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
```

**API endpoints:**
- `GET /api/gallery` — list builds (supports `?category=`, `?sort=newest|stars|parts`, `?q=search`)
- `POST /api/gallery` — save a build (accepts title, description_json, thumbnail_b64)
- `GET /api/gallery/:id` — get single build
- `PATCH /api/gallery/:id/star` — submit star rating (accepts `stars: 1-5`, computes running average)

**Backend files:**
- `backend/storage/gallery_db.py` — SQLite connection, CRUD operations
- `backend/api/routes_gallery.py` — FastAPI route handlers

**Frontend UI:**
- Card grid layout with thumbnails
- Each card: thumbnail, title, part count, star rating, category badge
- Filter by category dropdown, sort dropdown, search box
- "Build This" button on each card -> opens `/guide/:buildId`
- Star rating: click 1-5 stars inline

**Save flow:** On BuildSession page, after generation, a **"Save to Gallery"** button:
1. Captures 3D canvas as PNG via `canvas.toDataURL()`
2. POSTs to `/api/gallery` with title (derived from description.object), JSON, and thumbnail
3. Shows success toast

---

## 3. Navigation & Integration

### 3.1 Route Structure

| Route | Page | Status |
|-------|------|--------|
| `/` | Home | Existing |
| `/build` | BuildSession | Existing (path may change) |
| `/guide/:buildId` | Guidance Mode | New |
| `/explore` | Compare + Gallery | New |
| `/about` | About | Existing |

### 3.2 Navigation Bar

Persistent top nav added to all pages:
```
+-----------------------------------------------------+
| [brick icon] LegoGen    [Home] [Build] [Explore] [About] |
+-----------------------------------------------------+
```

Styled consistent with existing dark theme (bg-gray-950).

### 3.3 Feature Flow

```
Home -> "Get Started" -> Build Session
                             |
                      Generate build
                             |
                  +----------+----------+
                  v          v          v
           Save to       Start       View
           Gallery     Building    Validation
              |         (Guide)
              v            ^
           Explore         |
           Gallery --------+
          "Build This"
```

Key integration: every gallery build can be opened in guidance mode. During a demo:
1. Generate a build live (ML inference)
2. Save to gallery (persistence)
3. Open guidance mode (interaction design)
4. Switch to Compare tab (ML evaluation)

### 3.4 Changes to Existing Pages

**BuildSession** — add two buttons after build generation:
- "Save to Gallery" — stores the build
- "Start Building" — opens guidance mode

**No other changes to existing pages or components.**

---

## 4. New Files Summary

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/pages/GuidancePage.tsx` | Guidance mode page layout and state |
| `frontend/src/pages/ExplorePage.tsx` | Explore page with tab switching |
| `frontend/src/components/GuidanceViewer.tsx` | Enhanced 3D viewer for guidance |
| `frontend/src/components/BrickMesh.tsx` | Shared brick rendering (extracted) |
| `frontend/src/components/StepControls.tsx` | Prev/Play/Next controls + timer |
| `frontend/src/components/PartsChecklist.tsx` | Interactive parts checklist |
| `frontend/src/components/VoiceNarrator.ts` | Web Speech API wrapper |
| `frontend/src/components/CompareTab.tsx` | Model comparison UI |
| `frontend/src/components/GalleryTab.tsx` | Gallery grid + cards |
| `frontend/src/components/GalleryCard.tsx` | Individual gallery card |
| `frontend/src/components/MetricsBar.tsx` | CSS bar chart for metrics |
| `frontend/src/components/NavBar.tsx` | Top navigation bar |
| `frontend/src/api/legogen.ts` | Extended with gallery + comparison API calls |

### Backend
| File | Purpose |
|------|---------|
| `backend/api/routes_gallery.py` | Gallery CRUD endpoints |
| `backend/storage/gallery_db.py` | SQLite connection and queries |
| `scripts/precompute_comparisons.py` | Batch inference across checkpoints |

### Data
| Path | Purpose |
|------|---------|
| `data/comparisons/*.json` | Pre-computed comparison results |
| `data/gallery.db` | SQLite database (created at runtime) |

---

## 5. Technical Decisions

- **No auth system.** Single-user FYP — unnecessary complexity.
- **No CV brick detection.** Webcam is a reference view, not computer vision.
- **No charting library.** CSS bars are sufficient for metrics visualization.
- **No export (LDraw/PDF).** Out of scope.
- **SQLite, not Postgres.** Single-file DB, zero setup, perfect for FYP.
- **Web Speech API, not external TTS.** Browser-native, zero dependencies, works offline.
- **Pre-computed comparisons, not live.** Running 3 checkpoints x 20 inputs live is too slow for demo. Pre-compute once, serve as static JSON.
- **`aiosqlite` for async SQLite.** FastAPI is async, so DB calls should be non-blocking.

---

## 6. Out of Scope

- User authentication / multi-user
- Brick detection via computer vision
- Export to LDraw / PDF instruction booklet
- Real physics simulation
- Mobile-specific layout
- Deployment / hosting
