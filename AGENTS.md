# Exploratory scientific programming

- Keep numerical and data-flow code direct and locally readable. Prefer
  small repetition to premature abstraction. Extract helpers for actual
  reuse or error-prone logic, not hypothetical extensibility; do not add
  frameworks, configuration layers, or class hierarchies without a
  concrete need in the current task.
- Treat notebooks as source: include them in searches and read .ipynb
  cells directly, without nbconvert. Put all imports at the top of the
  file or notebook; in notebooks, place them under the first heading.
  No code may precede the first heading.
- Use H1 headings for top-level groups of steps, not for the notebook
  title. Headings and subheadings define executable cell groups: organize
  them around logically separate work or steps worth rerunning together
  during exploration. Keep each cell focused on one logical step, with
  cell-specific parameters/helpers near their use.
- Before changing a scientific path, trace its actual inputs and consumers.
  Check dataset identity, units, coordinate frames, time conventions,
  missing-data handling, and event definitions where relevant. Treat
  changes to these as scientific changes, not incidental cleanup.
- Do not add try/except unless explicitly requested. Let unexpected
  failures surface rather than producing plausible-looking partial results.
  Ask before an optimization bypasses or short-circuits existing functions.
- For changes affecting scientific results:
  1. Identify or construct a small representative case and keep its inputs
     and parameters fixed.
  2. Record the relevant existing outputs before editing.
  3. Make the change and rerun the same case through the affected workflow.
  4. Investigate unexpected changes, including coverage and missingness.
     For intended numerical changes, explain the expected difference and
     use physically meaningful tolerances, not blanket bitwise equality.
  Reuse relevant existing checks. Prefer checks of scientific outputs to
  test scaffolding tied to implementation details. State what actually ran
  and what remains unvalidated.
- Keep comparison runs separate from saved research results. Do not
  overwrite baselines or clear notebook outputs merely to tidy a change;
  saved outputs may be scientific evidence.
- Preface commits with "Codex: " or "Copilot: ".
- Prefer Jupyter rich display for dataframe/list previews. Do not add
  `.head()` to a cell's final preview object without a specific reason;
  this does not apply to `print()`.
- In notebooks, render symbolic math with MathJax (`display(Math(...))`).
  For symbolic expressions with more than three terms, prefer LaTeX
  rendering over plain-text prints.

- In Library/SW, keep native ACE and ACE-at-Earth distinct. Preserve the
  separation between the propagated cube and comparison-series corrections;
  check archive readers and writers together when changing their schema.
- Run code through Conda: prefer `conda run -n icme3.12-cuda ...`, then
  `conda run -n icme3.12-metal ...`, then `conda run -n icme3.12 ...`, using
  whichever environment exists. These are environment names, not executables.
