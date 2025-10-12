# Diagrams We Need to Create (Don't Have PNG For)

## DIAGRAM 1: Loan Lifecycle (Slide 2)

```mermaid
flowchart LR
    A[Loan Origination<br/>Static Features:<br/>CreditScore, DTI,<br/>InterestRate, LoanAmount] --> B[Monthly Performance<br/>Months 0-13<br/>UPB, Payments,<br/>Balance Tracking]
    B --> C{Loan Outcome}
    C -->|Normal| D[yi = 0]
    C -->|Anomaly| E[yi = 1<br/>Default/Missed Payment]

    style A fill:#4285F4,color:#fff
    style B fill:#FBBC04,color:#000
    style D fill:#34A853,color:#fff
    style E fill:#EA4335,color:#fff
```

---

## DIAGRAM 2: Final Model Architecture (Slide 21)

```mermaid
flowchart TD
    A[Raw Data<br/>30,504 × 145] --> B[Feature Builder Advanced]

    B --> B1[Sentinel Mapping]
    B1 --> B2[Temporal Engineering]
    B2 --> B3[Amortization Signals]
    B3 --> B4[Scaling + PCA 80]

    B4 --> C{Output}
    C --> C1[X_scaled 166]
    C --> C2[X_embed 80]

    C1 & C2 --> D[Detector Portfolio<br/>~20 detectors]

    D --> D1[LOF variants]
    D --> D2[Cluster-LOF]
    D --> D3[k-distance]
    D --> D4[IForest]
    D --> D5[Others]

    D1 & D2 & D3 & D4 & D5 --> E[Score Sets]
    E --> F[Train-CDF Calibration]
    F --> G[Select Top 10<br/>AUPRC ≥ 0.16]
    G --> H[Fusion<br/>max_rank_top2]
    H --> I[Anomaly Scores]

    style A fill:#4285F4,color:#fff
    style B fill:#34A853,color:#fff
    style D fill:#FBBC04,color:#000
    style H fill:#9C27B0,color:#fff
    style I fill:#EA4335,color:#fff
```

---

## DIAGRAM 3: LOF Intuition (Slide 13)

```mermaid
graph LR
    subgraph Normal
        A((●))
        B((●))
        C((●))
        D((●))
        E((●))
    end

    subgraph Sparse
        F((⚠️))
    end

    A -.-> F
    B -.-> F
    C -.-> F

    G[LOF = neighbor_density / point_density<br/>LOF ≈ 1 → Normal<br/>LOF >> 1 → Anomaly]

    style A fill:#34A853
    style B fill:#34A853
    style C fill:#34A853
    style D fill:#34A853
    style E fill:#34A853
    style F fill:#EA4335
```

---

## How to Use:

1. Go to https://mermaid.live
2. Paste each code block
3. Export as PNG or SVG
4. Insert into Google Slides

That's it. Done. 3 diagrams total.
