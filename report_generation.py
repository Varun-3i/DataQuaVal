from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def generate_detailed_report(df, detailed_scores_df, overall_score):
    try:
        # Define the metrics list
        metrics = ['Completeness', 'Uniqueness', 'Validity', 'Timeliness', 'Consistency', 'Accuracy', 'Reliability']
        
        html_content = []

        # General Styling for Data Quality Report
        html_content.append("""
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f7f8;
                color: #333;
            }
            h1, h2 {
                text-align: center;
                color: #2c3e50;
            }
            .container {
                margin: 20px auto;
                padding: 20px;
                max-width: 1200px;
                background: #fff;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 10px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }
            th {
                background-color: #2c3e50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .dropdown-container {
                margin: 20px auto;
                text-align: center;
            }
            select {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100%;
                max-width: 300px;
            }
        </style>
        """)

        # Section 1: Overall Quality Scores
        html_content.append("<div class='container'>")
        html_content.append(f"<h2>Overall Data Quality Score: {overall_score:.2f}%</h2>")

        # Section 2: Column-wise Quality Scores
        html_content.append("<h3>Detailed Column Quality Scores</h3>")
        html_content.append("<table>")
        html_content.append("<tr><th>Column</th>" + "".join(f"<th>{metric}</th>" for metric in metrics) + "</tr>")
        for col, scores in detailed_scores_df.iterrows():
            html_content.append("<tr>" + f"<td>{col}</td>" + "".join(f"<td>{scores.get(metric, 0):.2f}%</td>" for metric in metrics) + "</tr>")
        html_content.append("</table>")

        # Section 3: Overall Quality Metrics Table
        html_content.append("<h3>Overall Quality Scores</h3>")
        html_content.append("<table>")
        html_content.append("<tr><th>Metric</th><th>Average Score (%)</th></tr>")
        for metric in metrics:
            overall_metric_score = detailed_scores_df[metric].mean()
            html_content.append(f"<tr><td>{metric}</td><td>{overall_metric_score:.2f}%</td></tr>")
        html_content.append("</table>")

        html_content.append("</div>")  # Closing container for report

        # Visualizations Section
        html_content.append("<div class='visualizations-container'>")
        html_content.append("<h1>Column Visualizations</h1>")

        html_content.append("<div class='dropdown-container'>")
        html_content.append("<h2>Select a Column to View Visualizations</h2>")
        html_content.append("<select id='column-select' onchange='showColumnCharts(this.value)'>")
        html_content.append("<option value=''>Select a Column</option>")
        charts_data = {}

        for col, scores in detailed_scores_df.iterrows():
            html_content.append(f"<option value='{col}'>{col}</option>")
            values = [scores.get(metric, 0) for metric in metrics]

            # Bar Chart Generation
            plt.figure(figsize=(5, 3))
            plt.bar(metrics, values, color='skyblue')
            plt.title(f"Bar Chart for {col}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            bar_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()

            # Store Bar Chart Data Only
            charts_data[col] = {'bar_chart': bar_chart}

        html_content.append("</select>")
        html_content.append("</div>")  # Closing dropdown container

        # Charts Container
        html_content.append("<div class='charts-container' id='charts-container' style='display: none;'>")
        for col, charts in charts_data.items():
            html_content.append(f"""
            <div id='{col}-charts' class='chart-section' style='display: none;'>
                <h3>{col} - Bar Chart</h3>
                <div class="chart-container">
                    <img src='data:image/png;base64,{charts['bar_chart']}' alt='Bar Chart' />
                </div>
            </div>
            """)
        html_content.append("</div>")  # Closing charts container
        html_content.append("</div>")  # Closing visualizations container

        # JavaScript for Interactivity
        html_content.append("""
        <script>
            function showColumnCharts(column) {
                const chartSections = document.querySelectorAll('.chart-section');
                chartSections.forEach(section => section.style.display = 'none');
                if (column) {
                    const selectedSection = document.getElementById(`${column}-charts`);
                    if (selectedSection) {
                        document.getElementById('charts-container').style.display = 'block';
                        selectedSection.style.display = 'block';
                    }
                } else {
                    document.getElementById('charts-container').style.display = 'none';
                }
            }
        </script>
        """)

        # Additional Styling for Visualizations Section
        html_content.append("""
        <style>
            .visualizations-container {
                background-color: #f9f9f9;
                padding: 20px;
                margin-top: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 10px;
            }
            .charts-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-top: 20px;
            }
            .chart-container {
                width: 60%;
                text-align: center;
            }
        </style>
        """)

        return "\n".join(html_content)

    except Exception as e:
        print(f"Error generating report with dropdown visualizations: {e}")
        return ""



def generate_quality_summary(df, scores_df):
    try:
        # Initialize the HTML content with inline CSS for styling
        html_content = []
        html_content.append("""
        <style>
            /* General Reset and Body Styling */
            body {
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: #f4f7f8;
                color: #333;
            }

            h1 {
                text-align: center;
                color: #2c3e50;
                font-size: 2.5rem;
                margin: 20px 0;
            }

            /* Main Container Styling */
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }

            /* Flexbox Cards for Metrics */
            .metrics-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
            }

            .metric-card {
                flex: 1;
                min-width: 300px;
                background: linear-gradient(145deg, #ffffff, #e6e6e6);
                border-radius: 15px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease-in-out;
                padding: 20px;
                text-align: center;
            }

            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 4px 6px 15px rgba(0, 0, 0, 0.2);
            }

            .metric-title {
                font-size: 1.5rem;
                color: #007bff;
                margin-bottom: 15px;
                font-weight: bold;
            }

            .passing-percentage {
                font-size: 1.2rem;
                color: #28a745;
                font-weight: bold;
                margin-bottom: 10px;
            }

            /* List of Columns */
            .columns-list {
                list-style: none;
                padding: 0;
            }

            .columns-list li {
                padding: 8px;
                margin-bottom: 5px;
                background: #f0f9ff;
                color: #333;
                border-radius: 5px;
                transition: background 0.2s ease-in-out;
            }

            .columns-list li:hover {
                background: #007bff;
                color: #fff;
            }

            /* Footer for No Columns Passed */
            .no-columns {
                color: #ff0000;
                font-style: italic;
            }
        </style>

        <div class="container">
            <div class="metrics-container">
        """)

        # Generate HTML content for each metric card
        for metric in scores_df.columns:
            columns_passing = scores_df[scores_df[metric] >= 80].index.tolist()
            passing_percentage = (len(columns_passing) / len(scores_df)) * 100

            html_content.append(f"""
            <div class="metric-card">
                <div class="metric-title">{metric}</div>
                <div class="passing-percentage">{passing_percentage:.2f}% Passing</div>
            """)

            if columns_passing:
                html_content.append("<ul class='columns-list'>")
                for col in columns_passing:
                    html_content.append(f"<li>{col}</li>")
                html_content.append("</ul>")
            else:
                html_content.append("<p class='no-columns'>No columns passed this metric.</p>")

            html_content.append("</div>")  # Closing metric-card

        # Close containers
        html_content.append("""
            </div> <!-- Closing metrics-container -->
        </div> <!-- Closing container -->
        """)

        return "\n".join(html_content)

    except Exception as e:
        print(f"Error generating quality summary report: {e}")
        return ""

def generate_ydata_profiling_report(df, detailed_report_content, quality_summary_content, output_path="ydata_profiling_report.html"):
    try:
        # Generate the YData Profiling report
        profile = ProfileReport(df, title="YData Profiling Report", explorative=True)

        # Save the profiling report to a temporary file
        temp_path = "temp_report.html"
        profile.to_file(temp_path)

        with open(temp_path, "r", encoding="utf-8") as f:
            report_html = f.read()

        # Extract just the body content of the YData Profiling report (to avoid duplication)
        start_body = report_html.find("<body>") + len("<body>")
        end_body = report_html.find("</body>")
        profile_body_content = report_html[start_body:end_body]

        # Enhanced design with navbar and clean section transitions
        custom_sections = f"""
<style>
    /* General Reset */
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f7f9fc;
        color: #333;
        line-height: 1.6;
    }}

    /* Navbar Styling */
    .navbar {{
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: #fff;
        padding: 15px 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
    }}
    .navbar a {{
        color: #fff;
        text-decoration: none;
        font-weight: 500;
        margin: 0 15px;
        padding: 8px 12px;
        border-radius: 4px;
        transition: all 0.3s ease;
    }}
    .navbar a:hover {{
        background-color: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }}

    /* Section Container */
    .section-content {{
        display: none;
        margin: 30px auto;
        padding: 25px;
        max-width: 95%;
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-in-out;
    }}
    .section-content.active {{
        display: block;
    }}

    /* Section Titles */
    .section-title {{
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        color: #2575fc;
    }}

    /* Quality Summary Styling */
    .quality-summary {{
        text-align: center;
        padding: 30px;
        border-radius: 10px;
        background: #f6f8fa;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        font-size: 18px;
        color: #444;
        margin-top: 20px;
    }}
    .quality-summary h2 {{
        margin-bottom: 15px;
        color: #6a11cb;
        font-weight: bold;
    }}

    /* Animations */
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(10px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
</style>

<!-- Navbar -->
<div class="navbar">
    <a href="#" onclick="showSection('overview')">Overview</a>
    <a href="#" onclick="showSection('detailed-report')">Detailed Quality Report</a>
    <a href="#" onclick="showSection('quality-summary')">Quality Summary</a>
</div>

<!-- Sections -->
<div id="overview" class="section-content active">
    {profile_body_content}
</div>
<div id="detailed-report" class="section-content">
    <h2 class="section-title">Detailed Quality Report</h2>
    {detailed_report_content}
</div>
<div id="quality-summary" class="section-content quality-summary">
    <h2>Quality Summary</h2>
    {quality_summary_content}
</div>

<script>
    function showSection(sectionId) {{
        const sections = document.querySelectorAll('.section-content');
        sections.forEach(section => section.classList.remove('active'));
        document.getElementById(sectionId).classList.add('active');
    }}
</script>
"""

        # Replace <body> tag to include the custom sections
        report_html = report_html[:start_body] + custom_sections + report_html[end_body:]

        # Write the final report to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        print(f"Enhanced YData Profiling Report generated successfully: {output_path}")
    except Exception as e:
        print(f"Error generating YData profiling report: {e}")
