# Competitive Landscape Generator (CLG)

A hybrid **LLM + TF-IDF + public market data** pipeline for generating a **structured, strategy-grade competitive landscape** for any company.

CLG classifies related companies into **strategic relationship categories** such as:

- Direct competitors  
- Adjacent competitors  
- Substitutes  
- Suppliers  

This allows you to move from “who looks similar?” to “who matters strategically, and why.”


## Overview

**Competitive Landscape Generator (CLG)** produces a **structured competitive map** for any target company (public, private, or fictional) using a hybrid approach combining:

- **OpenAI LLMs** for semantic understanding and strategic reasoning  
- **FMP API** for ticker resolution and enrichment with public company data  
- **TF-IDF cosine similarity** for deterministic business-model similarity  
- **Rule-based and LLM validation** for strategic relationship classification  

### Outputs

- `competitive_landscape.csv`  
- `competitive_landscape.parquet`  
- `target_business_profile.json`  

Each company is classified into one of:

- `direct_competitor`
- `adjacent_competitor`
- `substitute`
- `supplier`
- `downstream_customer`
- `unrelated` (filtered out of final output)



## Example Output

```text
COMPETITIVE LANDSCAPE (STRATEGIC VIEW)
============================================================
Direct Competitors:
  - BigCommerce Holdings, Inc. (BIGC, NASDAQ)
  - Squarespace, Inc. (SQSP, NYSE)
  - Wix.com Ltd. (WIX, NASDAQ)

Adjacent Competitors:
  - Lightspeed Commerce Inc. (LSPD.TO, TSX)
  - Toast, Inc. (TOST, NYSE)

Substitutes:
  - Etsy, Inc. (ETSY, NYSE)

Suppliers:
  - Adyen N.V. (ADYYF, OTC)
  - Cloudflare, Inc. (NET, NYSE)
```

## Architecture
                ┌──────────────────────────┐
                │        TARGET INPUT      │
                │ name / desc / url / SIC  │
                └─────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ 1. Build TARGET profile │
                 │  - products/services    │
                 │  - customers            │
                 │  - jobs-to-be-done      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ 2. LLM candidate lists  │
                 │   (per strategic role)  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ 3. Resolve tickers via  │
                 │    FMP search API       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ 4. Enrich using FMP     │
                 │    company profiles     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ 5. Score proximity      │
                 │  TF-IDF (70%)           │
                 │  Industry sim (30%)     │
                 └────────────┬────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │ 6. LLM strategic classification│
              └───────────────┬────────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │ 7. Final grouped landscape     │
              │    CSV + Parquet + Print       │
              └────────────────────────────────┘


## Installation
### 1. Install required packages

```bash
pip install \
    requests \
    pandas \
    scikit-learn \
    pyarrow \
    python-dotenv \
    openai \
    pydantic
```

### 2. Set environment variables

Create a .env file or export manually:

```bash
OPENAI_API_KEY=your_openai_key
FMP_API_KEY=your_fmp_key
```

## Usage

Run from the command line:

```bash
python Competitive_landscape_pipeline.py \
  --name "Shopify Inc." \
  --desc "Shopify provides a cloud-based commerce platform..." \
  --sic "Computer Programming, Data Processing, and Other Computer Related Services" \
  --outdir ./runs/shopify
```
## Arguments

| Argument      | Required | Description                                 |
| ------------- | -------- | ------------------------------------------- |
| `--name`      | Yes      | Target company name                         |
| `--desc`      | Yes      | Business description                        |
| `--url`       | Optional | Homepage URL                                |
| `--sic`       | Optional | Primary SIC classification (text)           |
| `--outdir`    | Optional | Output directory (default: `./runs/output`) |
| `--max_final` | Optional | Maximum final entities (3–10)               |


## Output Files

```bash
/runs/<run_name>/
    ├── competitive_landscape.csv
    ├── competitive_landscape.parquet
    └── target_business_profile.json
```

## What This Is (and Is Not)

**This is:**

- A structured way to reason about competitive positioning
- A decision-support tool for strategy, research, and market analysis
- A hybrid system combining deterministic and generative methods

**This is not:**

- A valuation model
- A financial forecasting system
- A recommendation engine for investments

## Author

Developed by **Fathmat Samira Bakayoko**

Feel free to connect or reach out for discussion, collaboration, or feedback.
