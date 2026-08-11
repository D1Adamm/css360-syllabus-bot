# Illustrations

Custom illustration assets for Syllabus Model Lab. **This directory is
intentionally empty of artwork right now** — every slot falls back to a
geometric placeholder drawn from the design tokens, so the UI is complete and
correctly laid out without any of these files existing.

## Slots

| Name | Where it appears | Intent |
|---|---|---|
| `landing` | Public / landing experience | Education + syllabus themed. Sets the academic tone. |
| `empty-course` | A course with no syllabus or no content | Document / syllabus themed. |
| `contribute` | Student contribution intro or empty state | Document → question. |
| `model-ready` | Course model finished preparing | Small, quiet celebration. Not confetti. |

## Adding an asset

1. Add the file here, named after its slot (`landing.svg`).
2. Import it in `index.ts` and replace that slot's `null` with the import.
3. Update `ILLUSTRATION_ALT` if the default alt text no longer describes it.

Nothing else needs to change — `<Illustration name="landing" />` picks it up.

## Art direction

- SVG strongly preferred; it scales and stays crisp in both the small
  (empty-state) and large (landing) sizes.
- Use the palette in `src/styles/tokens.css`: purple as the structural colour,
  gold sparingly as an accent, warm neutrals for everything else.
- Academic, calm, and approachable. Line-led or flat shapes.
- **Avoid:** generic AI sparkles and glowing orbs, stock photography, neon
  gradients, glassmorphism, 3D renders, and anything implying this is an
  official University of Washington service.
- Keep artwork decorative. Never encode information that appears nowhere else,
  because the placeholder fallback cannot convey it.
- Test each asset against both `--surface` (white) and `--canvas` backgrounds.
