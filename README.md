# RL-Based Intelligent University Timetable Scheduler
**CT-469 Reinforcement Learning — Course Project**

University timetable scheduling is an NP-hard combinatorial optimisation problem. This project uses a hybrid GA + Dueling Double DQN approach to produce conflict-minimised schedules at scale (1200 students, 30 courses, 12 rooms), deployed as an interactive Gradio web application.

---

## How It Works

The system runs in three stages:

1. **Genetic Algorithm** generates an initial population of schedules and evolves them over multiple generations to produce a reasonable seed schedule.
2. **Dueling Double DQN** takes that seed and refines it by learning to reassign courses to better time slots and rooms, guided by a penalty-based reward signal.
3. **Constraint Propagation** validates the final schedule against hard and soft constraints after every step.

### MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | `[current_penalty, hard_violations, step_n, best_penalty]` — all normalised |
| **Action** | Reassign one course to a new (slot, room) pair — 10,800 possible actions |
| **Reward** | Clipped penalty improvement `± 1.0`, with a `+2.0` bonus for eliminating all hard violations |

### Constraints

**Hard (high penalty — must not be violated):**
- Room capacity overflow
- Teacher double-booked in the same slot
- Two courses in the same room at the same time
- Student enrolled in two simultaneous classes

**Soft (lower penalty — preferred):**
- No student with 3+ classes on the same day
- No courses in the evening (15:00) slot
- Teacher scheduled in their preferred time slots

---

## Baselines Compared

| Method | Description |
|--------|-------------|
| Manual | Random slot and room assignment |
| Greedy | Incrementally picks the lowest-penalty (slot, room) for each course |
| GA-only | Genetic algorithm run to completion, no DQN refinement |
| RL-Hybrid | GA seed + Dueling Double DQN refinement |

---

## Notebook Structure

| Cell | Contents |
|------|----------|
| 1 | Install dependencies |
| 2 | Google Drive mount and checkpoint paths |
| 3 | Imports and random seed |
| 4 | Data setup — students, courses, rooms, teacher preferences |
| 5 | Constraint evaluation functions (hard + soft, with breakdown) |
| 6 | Baseline 1: Manual (random) scheduler |
| 7 | Baseline 2: Greedy scheduler |
| 8 | Baseline 3: Genetic Algorithm |
| 9 | RL environment (MDP — state, action, reward, step) |
| 10 | Dueling DQN model and Replay Buffer |
| 11 | Training loop with Drive checkpoints every 10 episodes |
| 12 | Full pipeline runner — smart load from Drive if already trained |
| 13 | Scalability test — benchmarks all 4 methods at full scale |
| 14 | Analytics — training curves, comparison chart, heatmaps, conflict report |
| 15 | Gradio web application |
| 16 | App launch |

---

## Running the Project

### First Run (trains the model)

Run all cells top to bottom. Cell 12 will detect no saved results and run the full pipeline — Manual → Greedy → GA → RL-Hybrid — saving `best_model.pth` and all results to Google Drive. Takes roughly 5–10 minutes on Colab CPU.

### Subsequent Runs

Cell 12 detects existing Drive files and skips training. Jump straight to Cell 15–16 to launch the app. The model loads your previously trained weights automatically.

---

## Gradio App

The web app has 9 tabs:

| Tab | What it does |
|-----|--------------|
| 📊 Comparison | Bar charts comparing all 4 methods on penalty and hard violations |
| 📅 Schedule Viewer | Full timetable with optional conflict-only filter |
| 🌡 Density Heatmaps | Class density and student load across the week |
| 📉 Training Curves | DQN penalty, reward, loss, and epsilon over episodes |
| 👩‍🏫 Teacher Preferences | Set and save preferred time slots per teacher |
| 🏫 Courses & Rooms | Edit teacher assignments, enrolment sizes, room capacities |
| ⚙️ Custom Run | Configure GA and DQN parameters, run the full scheduler |
| 🎓 Student Lookup | View any student's timetable and flag time clashes |
| 🚪 Room Availability | Booking grid showing free and occupied slots per room |

When you press **▶ Run** in the Custom Run tab, the app runs all 4 algorithms (Manual, Greedy, GA-only, RL-Hybrid) so the Comparison tab always reflects the current settings.

---

## Deployment

### Colab (demo day)
Cell 16 uses `share=True` to generate a public `*.gradio.live` URL valid for 72 hours while the Colab tab stays open.

### Hugging Face Spaces (permanent)
1. Create a new Space with SDK: Gradio
2. Upload `app.py` (Cells 3–15 combined), `requirements.txt`, and `best_model.pth`
3. The Space builds automatically and stays live at `https://huggingface.co/spaces/YOUR_USERNAME/rl-timetable`

`requirements.txt`:
```
gradio
torch
numpy
pandas
matplotlib
seaborn
```

### Local
```bash
pip install gradio torch numpy pandas matplotlib seaborn
python app.py
```
Opens at `http://localhost:7860`.

---

## Dependencies

```
gradio
torch
numpy
pandas
matplotlib
seaborn
```

---

## Project Info

- **Course:** CT-469 Reinforcement Learning
- **Architecture:** Dueling Double DQN with experience replay
- **Scale:** 1200 students · 30 courses · 12 rooms · 30 time slots
- **Framework:** PyTorch · Gradio
