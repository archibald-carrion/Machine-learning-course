# Multiple Regression Cheat Sheet

## What is Multiple Regression?

**Multiple Regression** is a statistical method used to model the relationship between a **dependent variable** (the thing you want to predict) and **two or more independent variables** (the factors you think influence it).

Think of it like this:
> "I want to predict **house prices** based on **size**, **location**, and **number of rooms**."

Instead of just one input,we are using multiple inputs to get a more accurate prediction.

---

## Why Use Multiple Regression?

- To **analyze** how multiple factors affect an outcome.
- To **predict** a value more accurately.
- To **understand** the strength and form of relationships between variables.
- To **control** for confounding variables (avoid misleading results).

---

## How Does It Work?

The basic formula looks like:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

Where:
- `Y` = dependent variable (what you're predicting)
- `X₁, X₂, ..., Xₙ` = independent variables (the inputs/features)
- `β₀` = intercept (value of Y when all X's are 0)
- `β₁, β₂, ..., βₙ` = coefficients (how much each X influences Y)
- `ε` = error term (random noise we can't explain)

Each **coefficient** tells you:
- Positive value ➔ the variable increases the prediction.
- Negative value ➔ the variable decreases the prediction.

---

## Example

Imagine you are predicting a student's final grade based on:
- Hours studied (`X₁`)
- Attendance rate (`X₂`)
- Number of homework assignments completed (`X₃`)

The multiple regression might look like:

```
FinalGrade = 50 + 5*(HoursStudied) + 2*(AttendanceRate) + 1*(HomeworkCompleted) + ε
```

---

## When to Use Multiple Regression?

- When **you have more than one predictor**.
- When you want to **explain** or **predict** something complex that can't be done with a single variable.
- When relationships between variables are **linear** (or can be made approximately linear).

---

## Quick Tips

- **More variables ≠ better** model. Always check if adding a variable truly improves prediction, not just noise and complexity.
- **Check the p-values** of each coefficient to see if it's statistically significant, else it might not be worth including as it can get costly in terms of computation.
- **Visualize residuals** to catch patterns you might miss.
