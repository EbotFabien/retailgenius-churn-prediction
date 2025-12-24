const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        Header, Footer, AlignmentType, LevelFormat, 
        HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

function headerCell(text, width = 3120) {
    return new TableCell({
        borders: cellBorders, width: { size: width, type: WidthType.DXA },
        shading: { fill: "1E3A5F", type: ShadingType.CLEAR },
        children: [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 22 })] })]
    });
}

function dataCell(text, width = 3120) {
    return new TableCell({
        borders: cellBorders, width: { size: width, type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text, size: 22 })] })]
    });
}

function bullet(boldText, normalText) {
    return new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun({ text: boldText, bold: true }), new TextRun(normalText)] });
}

function numbered(boldText, normalText, ref = "numbered-list") {
    return new Paragraph({ numbering: { reference: ref, level: 0 },
        children: [new TextRun({ text: boldText, bold: true }), new TextRun(normalText)] });
}

const doc = new Document({
    styles: {
        default: { document: { run: { font: "Arial", size: 24 } } },
        paragraphStyles: [
            { id: "Title", name: "Title", basedOn: "Normal", run: { size: 56, bold: true, color: "1E3A5F", font: "Arial" }, paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER } },
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, color: "1E3A5F", font: "Arial" }, paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, color: "2E5A8F", font: "Arial" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
            { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, color: "3E7ABF", font: "Arial" }, paragraph: { spacing: { before: 180, after: 90 }, outlineLevel: 2 } }
        ]
    },
    numbering: {
        config: [
            { reference: "bullet-list", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
            { reference: "numbered-list", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
            { reference: "numbered-list-2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
            { reference: "numbered-list-3", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
        ]
    },
    sections: [{
        properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "RetailGenius Churn Prediction | EPITA AI PM 2025-2026", size: 18, color: "666666" })] })] }) },
        footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 }), new TextRun({ text: " of ", size: 18 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18 })] })] }) },
        children: [
            // TITLE PAGE
            new Paragraph({ spacing: { before: 2000 } }),
            new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("RetailGenius Customer Churn Prediction")] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 }, children: [new TextRun({ text: "AI Project Methodology - Graded Project Part 1", size: 28, color: "666666" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 }, children: [new TextRun({ text: "Functional Methodologies Report", size: 32, bold: true })] }),
            new Paragraph({ spacing: { before: 1500 } }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "EPITA International Programs | 2025-2026", size: 24 })] }),
            new Paragraph({ spacing: { before: 800 } }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Team Members: [Add Names]", size: 22, bold: true })] }),
            new Paragraph({ children: [new PageBreak()] }),

            // 1. PROJECT STRATEGY
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Project Strategy")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.1 Strategic Objectives")] }),
            new Paragraph({ children: [new TextRun("The AI-driven churn prediction project for RetailGenius aims to achieve:")] }),
            bullet("Proactive Customer Retention: ", "Identify at-risk customers before they churn, enabling targeted retention campaigns"),
            bullet("Revenue Protection: ", "Reduce revenue loss from customer attrition through early intervention"),
            bullet("Data-Driven Decision Making: ", "Leverage existing customer data for actionable insights"),
            bullet("Improved Customer Experience: ", "Personalize engagement based on churn risk profiles"),
            bullet("Operational Efficiency: ", "Optimize marketing spend by focusing on high-risk segments"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.2 Key Performance Indicators (KPIs)")] }),
            new Table({ columnWidths: [3000, 2500, 3860], rows: [
                new TableRow({ tableHeader: true, children: [headerCell("KPI", 3000), headerCell("Target", 2500), headerCell("Measurement", 3860)] }),
                new TableRow({ children: [dataCell("Model Accuracy", 3000), dataCell("> 85%", 2500), dataCell("Correctly classified / Total", 3860)] }),
                new TableRow({ children: [dataCell("Precision", 3000), dataCell("> 80%", 2500), dataCell("True positives / Predicted positives", 3860)] }),
                new TableRow({ children: [dataCell("Recall", 3000), dataCell("> 75%", 2500), dataCell("True positives / Actual positives", 3860)] }),
                new TableRow({ children: [dataCell("F1-Score", 3000), dataCell("> 78%", 2500), dataCell("Harmonic mean of Precision/Recall", 3860)] }),
                new TableRow({ children: [dataCell("ROC-AUC", 3000), dataCell("> 0.85", 2500), dataCell("Area under ROC curve", 3860)] }),
                new TableRow({ children: [dataCell("Churn Rate Reduction", 3000), dataCell("20% decrease", 2500), dataCell("Compare pre/post implementation", 3860)] }),
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("1.3 AI Contribution to Customer Retention")] }),
            bullet("Early Warning System: ", "Detect churn signals weeks before customers leave"),
            bullet("Customer Segmentation: ", "Group customers by churn risk for tailored interventions"),
            bullet("Personalized Offers: ", "AI-driven recommendations based on customer preferences"),
            bullet("Root Cause Analysis: ", "SHAP explanations identify why customers are likely to churn"),
            new Paragraph({ children: [new PageBreak()] }),

            // 2. PROJECT DESIGN - DATA
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Project Design")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.1 Data Sources")] }),
            new Table({ columnWidths: [2800, 3200, 3360], rows: [
                new TableRow({ tableHeader: true, children: [headerCell("Source", 2800), headerCell("Data Type", 3200), headerCell("Usage", 3360)] }),
                new TableRow({ children: [dataCell("User Interactions", 2800), dataCell("Views, searches, purchases, reviews", 3200), dataCell("Behavior patterns", 3360)] }),
                new TableRow({ children: [dataCell("Customer Demographics", 2800), dataCell("Age, location, tenure", 3200), dataCell("Segmentation", 3360)] }),
                new TableRow({ children: [dataCell("Order History", 2800), dataCell("Purchases, returns, frequency", 3200), dataCell("Transaction patterns", 3360)] }),
                new TableRow({ children: [dataCell("Customer Support", 2800), dataCell("Complaints, tickets, satisfaction", 3200), dataCell("Service quality indicators", 3360)] }),
                new TableRow({ children: [dataCell("Marketing Data", 2800), dataCell("Campaign responses, coupons", 3200), dataCell("Engagement metrics", 3360)] }),
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("2.2 Data Challenges")] }),
            bullet("Data Silos: ", "Integration required across multiple systems (CRM, ERP, marketing)"),
            bullet("Data Quality: ", "Missing values, inconsistencies, and outliers need handling"),
            bullet("Privacy & Security: ", "GDPR compliance, encryption, access controls required"),
            bullet("Feature Engineering: ", "Deriving meaningful features from raw transactional data"),
            bullet("Class Imbalance: ", "Churners typically represent a small percentage of customers"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.3 AI Models")] }),
            new Paragraph({ children: [new TextRun("Suitable models for churn prediction:")] }),
            new Table({ columnWidths: [2500, 3500, 3360], rows: [
                new TableRow({ tableHeader: true, children: [headerCell("Model", 2500), headerCell("Strengths", 3500), headerCell("Use Case", 3360)] }),
                new TableRow({ children: [dataCell("Random Forest", 2500), dataCell("Robust, handles non-linearity, interpretable", 3500), dataCell("Primary model", 3360)] }),
                new TableRow({ children: [dataCell("XGBoost", 2500), dataCell("High accuracy, handles imbalance well", 3500), dataCell("Comparison model", 3360)] }),
                new TableRow({ children: [dataCell("LightGBM", 2500), dataCell("Fast training, memory efficient", 3500), dataCell("Large datasets", 3360)] }),
                new TableRow({ children: [dataCell("Logistic Regression", 2500), dataCell("Interpretable, baseline performance", 3500), dataCell("Baseline comparison", 3360)] }),
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("2.4 Model Training & Validation")] }),
            numbered("Data Split: ", "80% training, 20% testing with stratified sampling", "numbered-list"),
            numbered("Cross-Validation: ", "5-fold cross-validation for hyperparameter tuning", "numbered-list"),
            numbered("Metrics: ", "Optimize for F1-score due to class imbalance", "numbered-list"),
            numbered("Validation: ", "Hold-out test set for final evaluation", "numbered-list"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.5 Model Versioning & Serving")] }),
            bullet("MLflow Tracking: ", "Log parameters, metrics, and artifacts for each experiment"),
            bullet("Model Registry: ", "Version control with staging/production environments"),
            bullet("Model Serving: ", "REST API endpoint via MLflow serving for real-time predictions"),
            new Paragraph({ children: [new PageBreak()] }),

            // DEPLOYMENT & MONITORING
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.6 Deployment Strategies")] }),
            new Table({ columnWidths: [2500, 6860], rows: [
                new TableRow({ tableHeader: true, children: [headerCell("Strategy", 2500), headerCell("Description", 6860)] }),
                new TableRow({ children: [dataCell("Shadow Mode", 2500), dataCell("Run model alongside existing systems without affecting decisions", 6860)] }),
                new TableRow({ children: [dataCell("A/B Testing", 2500), dataCell("Compare model-driven retention vs. traditional approaches", 6860)] }),
                new TableRow({ children: [dataCell("Canary Deployment", 2500), dataCell("Gradual rollout to subset of customers before full deployment", 6860)] }),
                new TableRow({ children: [dataCell("Blue-Green", 2500), dataCell("Maintain two environments for zero-downtime updates", 6860)] }),
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("2.7 Production Considerations")] }),
            bullet("Infrastructure: ", "Scalable cloud deployment (AWS/GCP/Azure) with auto-scaling"),
            bullet("Latency: ", "Sub-second prediction response time for real-time scoring"),
            bullet("Security: ", "API authentication, data encryption, audit logging"),
            bullet("Integration: ", "REST API integration with CRM and marketing automation systems"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.8 Monitoring Strategy")] }),
            bullet("Model Performance: ", "Track accuracy, precision, recall weekly on new predictions"),
            bullet("Data Drift: ", "Monitor input feature distributions for significant changes"),
            bullet("Model Drift: ", "Compare predicted vs. actual churn rates monthly"),
            bullet("Retraining Triggers: ", "Automatic alerts when performance degrades below thresholds"),
            bullet("Dashboard: ", "Real-time monitoring via MLflow UI and custom dashboards"),
            new Paragraph({ children: [new PageBreak()] }),

            // 3. PROJECT TEAM
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Project Team")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.1 Team Roles & Responsibilities")] }),
            new Table({ columnWidths: [2500, 3500, 3360], rows: [
                new TableRow({ tableHeader: true, children: [headerCell("Role", 2500), headerCell("Responsibilities", 3500), headerCell("Skills Required", 3360)] }),
                new TableRow({ children: [dataCell("Project Manager", 2500), dataCell("Timeline, resources, stakeholder communication", 3500), dataCell("Agile, risk management", 3360)] }),
                new TableRow({ children: [dataCell("Data Engineer", 2500), dataCell("Data pipelines, ETL, infrastructure", 3500), dataCell("SQL, Spark, cloud platforms", 3360)] }),
                new TableRow({ children: [dataCell("Data Scientist", 2500), dataCell("Model development, feature engineering", 3500), dataCell("Python, ML, statistics", 3360)] }),
                new TableRow({ children: [dataCell("ML Engineer", 2500), dataCell("Model deployment, MLOps, monitoring", 3500), dataCell("MLflow, Docker, APIs", 3360)] }),
                new TableRow({ children: [dataCell("Business Analyst", 2500), dataCell("Requirements, KPIs, business context", 3500), dataCell("Domain knowledge, analytics", 3360)] }),
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("3.2 Cross-Functional Collaboration")] }),
            bullet("Marketing Team: ", "Integrate predictions into campaign targeting systems"),
            bullet("Customer Support: ", "Alert agents about high-risk customers for proactive outreach"),
            bullet("Product Team: ", "Use churn insights to improve product features"),
            bullet("Finance: ", "ROI tracking and budget allocation for retention programs"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.3 Team Alignment")] }),
            bullet("Sprint Reviews: ", "Bi-weekly demos to align on progress and priorities"),
            bullet("Shared OKRs: ", "Team objectives linked to business KPIs"),
            bullet("Documentation: ", "Confluence/Notion for shared knowledge base"),
            bullet("Communication: ", "Daily standups, Slack channels for async collaboration"),
            new Paragraph({ children: [new PageBreak()] }),

            // 4. GOVERNANCE & COMMUNICATION
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Project Governance & Communication")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1 Key Stakeholders")] }),
            new Table({ columnWidths: [2500, 3500, 3360], rows: [
                new TableRow({ tableHeader: true, children: [headerCell("Stakeholder", 2500), headerCell("Interest", 3500), headerCell("Communication", 3360)] }),
                new TableRow({ children: [dataCell("Executive Sponsor", 2500), dataCell("ROI, strategic alignment", 3500), dataCell("Monthly steering committee", 3360)] }),
                new TableRow({ children: [dataCell("Marketing Director", 2500), dataCell("Campaign effectiveness", 3500), dataCell("Weekly updates", 3360)] }),
                new TableRow({ children: [dataCell("CTO/Tech Lead", 2500), dataCell("Technical feasibility, integration", 3500), dataCell("Sprint reviews", 3360)] }),
                new TableRow({ children: [dataCell("Data Privacy Officer", 2500), dataCell("Compliance, GDPR", 3500), dataCell("Quarterly audits", 3360)] }),
                new TableRow({ children: [dataCell("Customer Success", 2500), dataCell("Retention outcomes", 3500), dataCell("Bi-weekly syncs", 3360)] }),
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("4.2 Governance Instances")] }),
            bullet("Steering Committee: ", "Monthly executive review of progress, risks, and decisions"),
            bullet("Technical Review Board: ", "Bi-weekly architecture and design decisions"),
            bullet("Sprint Planning: ", "Bi-weekly backlog grooming and sprint planning"),
            bullet("Daily Standups: ", "15-minute daily sync on blockers and progress"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.3 Communication Plan")] }),
            new Paragraph({ children: [new TextRun({ text: "For Technical Teams:", bold: true })] }),
            bullet("", "Model metrics dashboards, Jupyter notebooks, technical documentation"),
            bullet("", "MLflow UI for experiment tracking and model comparison"),
            new Paragraph({ children: [new TextRun({ text: "For Non-Technical Teams:", bold: true })] }),
            bullet("", "Executive summaries with business impact metrics"),
            bullet("", "Visual dashboards showing churn predictions and retention outcomes"),
            bullet("", "Explainable AI outputs (SHAP) translated to business insights"),
            new Paragraph({ children: [new PageBreak()] }),

            // 5. PROJECT MANAGEMENT METHODOLOGY
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. AI Project Management Methodology")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.1 Methodology Choice: Agile/Scrum with CRISP-DM")] }),
            new Paragraph({ children: [new TextRun("We recommend a hybrid approach combining Agile/Scrum with CRISP-DM (Cross-Industry Standard Process for Data Mining):")] }),
            new Paragraph({ children: [new TextRun({ text: "Why this approach:", bold: true })] }),
            bullet("Iterative Development: ", "AI projects require experimentation and iteration"),
            bullet("Flexibility: ", "Adapt to changing requirements and model insights"),
            bullet("Stakeholder Engagement: ", "Regular demos keep business aligned"),
            bullet("Risk Management: ", "Early detection of issues through short sprints"),
            bullet("CRISP-DM Structure: ", "Provides ML-specific phases within Agile framework"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("5.2 Risk Management")] }),
            new Table({ columnWidths: [2800, 2800, 3760], rows: [
                new TableRow({ tableHeader: true, children: [headerCell("Risk", 2800), headerCell("Impact", 2800), headerCell("Mitigation", 3760)] }),
                new TableRow({ children: [dataCell("Poor Data Quality", 2800), dataCell("High - affects accuracy", 2800), dataCell("Early data profiling, validation pipelines", 3760)] }),
                new TableRow({ children: [dataCell("Model Underperformance", 2800), dataCell("High - no business value", 2800), dataCell("Baseline models, iterative improvement", 3760)] }),
                new TableRow({ children: [dataCell("Scope Creep", 2800), dataCell("Medium - delays", 2800), dataCell("Clear MVP definition, backlog prioritization", 3760)] }),
                new TableRow({ children: [dataCell("Integration Issues", 2800), dataCell("Medium - deployment delays", 2800), dataCell("Early API design, staging environment", 3760)] }),
                new TableRow({ children: [dataCell("Team Capacity", 2800), dataCell("Medium - velocity impact", 2800), dataCell("Cross-training, documentation", 3760)] }),
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 400 }, children: [new TextRun("5.3 Handling Costs & Planning Deviations")] }),
            bullet("Buffer Time: ", "20% buffer for ML experimentation and iteration cycles"),
            bullet("MVP Approach: ", "Deliver working model early, enhance incrementally"),
            bullet("Cost Tracking: ", "Monthly budget reviews tied to sprint deliverables"),
            bullet("Change Management: ", "Impact assessment for scope changes affecting timeline/budget"),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.4 Kanban Board (Trello)")] }),
            new Paragraph({ children: [new TextRun("A Trello board has been set up with the following columns:")] }),
            numbered("Backlog: ", "All project tasks and user stories", "numbered-list-2"),
            numbered("To Do: ", "Sprint-committed items", "numbered-list-2"),
            numbered("In Progress: ", "Currently being worked on", "numbered-list-2"),
            numbered("Review: ", "Awaiting code review or stakeholder approval", "numbered-list-2"),
            numbered("Done: ", "Completed and deployed", "numbered-list-2"),
            new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "Trello Board URL: ", bold: true }), new TextRun("[Insert your Trello board link here]")] }),
            new Paragraph({ children: [new PageBreak()] }),

            // REFERENCES
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("6. References")] }),
            numbered("Kaggle E-Commerce Churn Dataset: ", "https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction", "numbered-list-3"),
            numbered("MLflow Documentation: ", "https://mlflow.org/docs/latest/index.html", "numbered-list-3"),
            numbered("SHAP Documentation: ", "https://shap.readthedocs.io/", "numbered-list-3"),
            numbered("Cookiecutter Data Science: ", "https://drivendata.github.io/cookiecutter-data-science/", "numbered-list-3"),
            numbered("CRISP-DM Methodology: ", "https://www.datascience-pm.com/crisp-dm-2/", "numbered-list-3"),
            numbered("Claude AI: ", "Used for project guidance and documentation assistance", "numbered-list-3"),
        ]
    }]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("/home/claude/retailgenius-churn-prediction/reports/Part1_Functional_Methodologies_Report.docx", buffer);
    console.log("Part 1 Report created successfully!");
});
