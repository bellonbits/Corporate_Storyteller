from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import uuid
import datetime
import requests
from typing import List, Optional, Dict, Any
from fpdf import FPDF
import io
import textwrap
import matplotlib.ticker as mticker
import traceback

# Add corporate branding colors
CORPORATE_COLORS = {
    "primary": "#0F4C81",  # Deep blue
    "secondary": "#4A90E2",  # Light blue
    "accent": "#F2994A",  # Orange
    "success": "#6FCF97",  # Green
    "warning": "#F2C94C",  # Yellow
    "danger": "#EB5757",  # Red
    "neutral": "#606060",  # Gray
    "background": "#F8F9FA"  # Light gray
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configuration for Groq API
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

app = FastAPI(title="Corporate Data Storyteller", description="Transform business data into actionable insights with clear visualizations")

# Create necessary directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/visualizations", exist_ok=True)
os.makedirs("static/pdfs", exist_ok=True)
os.makedirs("tmp/uploads", exist_ok=True)  # Added missing directory
os.makedirs("tmp/pdfs", exist_ok=True)  # Added missing directory

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Corporate presentation styles based on industry
CORPORATE_STYLES = {
    "finance": {
        "name": "Financial Insights",
        "style": "Precise analysis with clear ROI and performance metrics",
        "intro": "The financial data reveals critical patterns that directly impact our bottom line...",
        "tone": "analytical, precise, actionable"
    },
    "tech": {
        "name": "Tech Performance",
        "style": "Forward-looking analysis with trend identification and innovation opportunities",
        "intro": "Our technology metrics show interesting patterns that can drive our next innovation cycle...",
        "tone": "innovative, future-focused, technical"
    },
    "marketing": {
        "name": "Market Analysis",
        "style": "Customer-focused insights with engagement and conversion highlights",
        "intro": "Customer behavior data indicates several opportunities to enhance our market position...",
        "tone": "engaging, customer-centric, strategic"
    },
    "operations": {
        "name": "Operational Excellence",
        "style": "Efficiency-focused analysis with process improvement recommendations",
        "intro": "Our operational metrics highlight key areas where efficiency gains can be realized...",
        "tone": "pragmatic, detailed, improvement-oriented"
    },
    "executive": {
        "name": "Executive Summary",
        "style": "High-level strategic insights with clear decision points",
        "intro": "The data presents a clear strategic picture that warrants executive attention...",
        "tone": "strategic, concise, decisive"
    }
}

# Model for story generation request
class StoryGenerationRequest(BaseModel):
    corporate_style: str
    business_context: str
    audience: str
    title: Optional[str] = None

# Model for API response
class StoryResponse(BaseModel):
    report_id: str
    title: str
    preview: str
    pdf_url: str

# Helper class for creating PDF documents with corporate branding
class CorporateReportPDF(FPDF):
    def __init__(self, title="Business Data Analysis", company_name="Your Company"):
        super().__init__()
        self.title = title
        self.company_name = company_name
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        
        # Add corporate header
        self.set_fill_color(15, 76, 129)  # Deep blue header
        self.rect(0, 0, 210, 30, style="F")
        self.set_text_color(255, 255, 255)  # White text
        self.set_font("Arial", "B", 24)
        self.cell(0, 15, self.title, ln=True, align="C")
        self.set_font("Arial", "I", 12)
        self.cell(0, 10, self.company_name, ln=True, align="C")
        
        # Reset text color for body
        self.set_text_color(0, 0, 0)
        self.ln(10)
        
    def section_title(self, title):
        self.set_font("Arial", "B", 16)
        self.set_fill_color(74, 144, 226)  # Light blue background
        self.set_text_color(255, 255, 255)  # White text
        self.cell(0, 10, title, ln=True, fill=True)
        self.set_text_color(0, 0, 0)  # Reset text color
        self.set_font("Arial", "", 11)
        self.ln(5)
        
    def section_body(self, body):
        self.set_font("Arial", "", 11)
        for line in textwrap.wrap(body, 80):
            self.cell(0, 6, line, ln=True)
        self.ln(5)
    
    def add_bullet_points(self, points):
        self.set_font("Arial", "", 11)
        for point in points:
            # Use a hyphen instead of bullet point for better compatibility
            self.cell(10, 6, "-", ln=0)
            wrapped_point = textwrap.wrap(point, 75)
            if wrapped_point:  # Check if there's any content to avoid empty lists
                self.cell(0, 6, wrapped_point[0], ln=True)
                for line in wrapped_point[1:]:
                    self.cell(10, 6, "", ln=0)
                    self.cell(0, 6, line, ln=True)
            else:
                self.cell(0, 6, "", ln=True)  # Empty line for empty points
        self.ln(5)
        
    def add_image(self, img_path, w=180):
        # Check if image exists to prevent errors
        if os.path.exists(img_path):
            self.image(img_path, x=15, y=None, w=w)
            self.ln(5)
        
    def image_caption(self, caption):
        self.set_font("Arial", "I", 9)
        self.cell(0, 6, caption, ln=True, align="C")
        self.set_font("Arial", "", 11)
        self.ln(10)
        
    def add_footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d')} | {self.company_name} Confidential", 0, 0, "C")

# Enhanced Data Analysis and Visualization Class for Corporate Context
class CorporateDataAnalyzer:
    def __init__(self, df, business_context=None):
        self.df = df
        self.business_context = business_context
        self.insights = []
        self.recommendations = []
        self.visualizations = []
        
    def detect_data_type(self):
        """Attempt to detect what type of business data we're working with"""
        columns = [col.lower() for col in self.df.columns]
        
        # Financial data detection
        financial_indicators = ['revenue', 'profit', 'sales', 'cost', 'margin', 'budget', 'expense', 'income', 'roi']
        financial_score = sum(any(fi in col for fi in financial_indicators) for col in columns)
        
        # Marketing data detection
        marketing_indicators = ['conversion', 'click', 'view', 'impression', 'campaign', 'lead', 'customer', 'acquisition']
        marketing_score = sum(any(mi in col for mi in marketing_indicators) for col in columns)
        
        # Operations data detection
        operations_indicators = ['time', 'efficiency', 'production', 'output', 'input', 'resource', 'utilization', 'capacity']
        operations_score = sum(any(oi in col for oi in operations_indicators) for col in columns)
        
        # HR data detection
        hr_indicators = ['employee', 'satisfaction', 'turnover', 'hire', 'retention', 'performance', 'salary', 'training']
        hr_score = sum(any(hi in col for hi in hr_indicators) for col in columns)
        
        scores = {
            'financial': financial_score,
            'marketing': marketing_score,
            'operations': operations_score,
            'hr': hr_score
        }
        
        # Return the highest scoring data type or 'general' if all scores are low
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        return 'general'
        
    def generate_business_insights(self):
        """Generate insights relevant to business context"""
        insights = []
        data_type = self.detect_data_type()
        
        # Basic statistics with business context
        basic_stats = {
            "type": "data_summary",
            "content": {
                "rows": int(len(self.df)),
                "columns": int(len(self.df.columns)),
                "time_period": self.detect_time_period(),
                "data_type": data_type,
                "key_metrics": self.identify_key_metrics(data_type)
            }
        }
        insights.append(basic_stats)
        
        # Growth/trend analysis for numeric columns
        growth_insights = self.analyze_growth_trends()
        if growth_insights:
            insights.append({
                "type": "growth_analysis",
                "content": growth_insights
            })
        
        # Find outliers and anomalies
        outliers = self.identify_outliers()
        if outliers:
            insights.append({
                "type": "outliers",
                "content": outliers
            })
        
        # Segment comparison if categorical columns exist
        segment_insights = self.compare_segments()
        if segment_insights:
            insights.append({
                "type": "segment_comparison",
                "content": segment_insights
            })
        
        # Generate specific recommendations based on insights
        self.generate_recommendations(insights, data_type)
        
        self.insights = insights
        return insights
    
    def detect_time_period(self):
        """Detect the time period covered in the dataset"""
        date_cols = [col for col in self.df.columns if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day'])]
        
        if date_cols:
            try:
                # Convert first date column to datetime
                date_col = date_cols[0]
                dates = pd.to_datetime(self.df[date_col], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) > 0:
                    start = valid_dates.min().strftime('%Y-%m-%d')
                    end = valid_dates.max().strftime('%Y-%m-%d')
                    duration_days = (valid_dates.max() - valid_dates.min()).days
                    
                    if duration_days <= 31:
                        period_type = "Monthly"
                    elif duration_days <= 92:
                        period_type = "Quarterly"
                    elif duration_days <= 366:
                        period_type = "Annual"
                    else:
                        period_type = "Multi-year"
                        
                    return {
                        "start_date": start,
                        "end_date": end,
                        "duration_days": duration_days,
                        "period_type": period_type
                    }
            except Exception as e:
                pass
                
        # Default if no time period detected
        return {"period_type": "Unknown"}
    
    def identify_key_metrics(self, data_type):
        """Identify the key metrics in the dataset based on data type"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        key_metrics = []
        
        # Common metrics by business area
        metric_keywords = {
            'financial': ['revenue', 'profit', 'margin', 'cost', 'sales', 'growth'],
            'marketing': ['conversion', 'ctr', 'cac', 'ltv', 'roi', 'engagement'],
            'operations': ['efficiency', 'output', 'utilization', 'time', 'rate'],
            'hr': ['turnover', 'satisfaction', 'performance', 'retention']
        }
        
        # Find metrics based on column names
        relevant_keywords = metric_keywords.get(data_type, metric_keywords['financial'])
        for col in numeric_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in relevant_keywords) or len(key_metrics) < 3:
                # Get basic stats for this metric
                if self.df[col].count() > 0:
                    key_metrics.append({
                        "name": col,
                        "mean": float(self.df[col].mean()),
                        "median": float(self.df[col].median()),
                        "min": float(self.df[col].min()),
                        "max": float(self.df[col].max()),
                        "std": float(self.df[col].std()),
                        "missing_pct": float(self.df[col].isna().mean() * 100)
                    })
        
        return key_metrics
    
    def analyze_growth_trends(self):
        """Analyze growth trends in numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        date_cols = [col for col in self.df.columns if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day'])]
        
        trends = {}
        
        # If we have date columns, analyze trends over time
        if date_cols and len(numeric_cols) > 0:
            try:
                date_col = date_cols[0]
                self.df['temp_date'] = pd.to_datetime(self.df[date_col], errors='coerce')
                
                # For each numeric column, calculate growth
                for col in numeric_cols[:3]:  # Limit to top 3 metrics
                    temp_df = self.df.dropna(subset=['temp_date', col]).sort_values('temp_date')
                    
                    if len(temp_df) > 1:
                        first_value = temp_df[col].iloc[0]
                        last_value = temp_df[col].iloc[-1]
                        
                        if first_value != 0:
                            pct_change = ((last_value - first_value) / first_value) * 100
                        else:
                            pct_change = 0
                            
                        # Determine trend direction
                        if pct_change > 5:
                            direction = "increasing"
                        elif pct_change < -5:
                            direction = "decreasing"
                        else:
                            direction = "stable"
                            
                        trends[col] = {
                            "start_value": float(first_value),
                            "end_value": float(last_value),
                            "pct_change": float(pct_change),
                            "direction": direction
                        }
                
                # Remove temp date column
                self.df.drop('temp_date', axis=1, inplace=True, errors='ignore')
                
            except Exception as e:
                pass
        
        return trends
    
    def identify_outliers(self):
        """Identify outliers in key metrics"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols[:3]:  # Focus on top 3 numeric columns
            # Calculate IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                pct_outliers = (outlier_count / len(self.df)) * 100
                
                # Only report if significant number of outliers
                if pct_outliers > 1:
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": float(pct_outliers),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "min_outlier": float(self.df.loc[outlier_mask, col].min()) if outlier_count > 0 else None,
                        "max_outlier": float(self.df.loc[outlier_mask, col].max()) if outlier_count > 0 else None
                    }
        
        return outliers
    
    def compare_segments(self):
        """Compare performance across different segments"""
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        segment_insights = {}
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Choose most promising categorical column (with reasonable cardinality)
            segment_col = None
            for col in categorical_cols:
                n_unique = self.df[col].nunique()
                if 2 <= n_unique <= 10:  # Reasonable number of segments
                    segment_col = col
                    break
            
            if segment_col:
                primary_metric = numeric_cols[0]
                segments = {}
                
                # Calculate metrics for each segment
                for segment_value, group in self.df.groupby(segment_col):
                    if not pd.isna(segment_value) and len(group) > 0:
                        segments[str(segment_value)] = {
                            "count": int(len(group)),
                            "percentage": float(len(group) / len(self.df) * 100),
                            "avg_value": float(group[primary_metric].mean()),
                            "median_value": float(group[primary_metric].median())
                        }
                
                if segments:
                    # Find best and worst performing segments
                    segment_performance = {k: v["avg_value"] for k, v in segments.items()}
                    best_segment = max(segment_performance, key=segment_performance.get)
                    worst_segment = min(segment_performance, key=segment_performance.get)
                    
                    segment_insights = {
                        "segment_column": segment_col,
                        "metric_analyzed": primary_metric,
                        "segment_details": segments,
                        "best_segment": best_segment,
                        "worst_segment": worst_segment,
                        "performance_gap": float(segment_performance[best_segment] - segment_performance[worst_segment])
                    }
        
        return segment_insights
    
    def generate_recommendations(self, insights, data_type):
        """Generate business recommendations based on insights"""
        recommendations = []
        
        # Process growth trends
        growth_data = next((item for item in insights if item["type"] == "growth_analysis"), None)
        if growth_data and growth_data["content"]:
            for metric, trend in growth_data["content"].items():
                if trend["direction"] == "increasing":
                    recommendations.append(f"Capitalize on the {trend['pct_change']:.1f}% growth in {metric} by identifying the key drivers and reinforcing those factors.")
                elif trend["direction"] == "decreasing":
                    recommendations.append(f"Address the {abs(trend['pct_change']):.1f}% decline in {metric} by investigating root causes and implementing corrective measures.")
        
        # Process outliers
        outlier_data = next((item for item in insights if item["type"] == "outliers"), None)
        if outlier_data and outlier_data["content"]:
            for metric, outlier_info in outlier_data["content"].items():
                recommendations.append(f"Investigate the {outlier_info['count']} outliers in {metric} ({outlier_info['percentage']:.1f}% of data) to identify anomalies that may represent risks or opportunities.")
        
        # Process segment comparison
        segment_data = next((item for item in insights if item["type"] == "segment_comparison"), None)
        if segment_data and segment_data["content"]:
            seg_info = segment_data["content"]
            metric = seg_info["metric_analyzed"]
            recommendations.append(f"Learn from the success of the {seg_info['best_segment']} segment and apply those lessons to improve performance in the underperforming {seg_info['worst_segment']} segment.")
            
        # Add data-type specific recommendations
        if data_type == "financial":
            recommendations.append("Consider performing a detailed profitability analysis by product/service line to identify optimization opportunities.")
        elif data_type == "marketing":
            recommendations.append("Analyze the customer journey touchpoints to identify conversion bottlenecks and optimization opportunities.")
        elif data_type == "operations":
            recommendations.append("Implement process mapping to identify efficiency improvement opportunities in the operational workflow.")
        elif data_type == "hr":
            recommendations.append("Conduct a deeper employee engagement analysis to identify factors affecting performance and retention.")
            
        self.recommendations = recommendations
        return recommendations
        
    def create_corporate_visualizations(self, base_path="static/visualizations", session_id=None):
        """Create business-focused visualizations following Knaflic's principles"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        viz_paths = []
        
        # Set style for visualizations - clean, minimal, professional
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a directory for this session's visualizations
        session_dir = os.path.join(base_path, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        try:
            # 1. Key Metrics Overview (clean bar chart)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                key_metrics = numeric_cols[:3]  # Focus on top 3 metrics
                
                plt.figure(figsize=(10, 6))
                avg_values = [self.df[col].mean() for col in key_metrics]
                
                # Create horizontal bar chart
                bars = plt.barh(key_metrics, avg_values, color=CORPORATE_COLORS["primary"])
                
                # Clean design
                plt.title("Key Metrics Overview", fontsize=14, fontweight='bold')
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.grid(axis='y', visible=False)
                
                # Add values to end of bars
                for i, v in enumerate(avg_values):
                    plt.text(v, i, f" {v:.1f}", va='center', fontweight='bold')
                
                plt.tight_layout()
                viz_path = os.path.join(session_dir, "key_metrics.png")
                plt.savefig(viz_path, dpi=300, bbox_inches="tight", facecolor='white')
                plt.close()
                
                viz_paths.append({
                    "path": viz_path,
                    "title": "Key Business Metrics",
                    "description": "Overview of the primary performance indicators in the dataset."
                })
            
            # 2. Trend Analysis (if time data available)
            date_cols = [col for col in self.df.columns if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day'])]
            if date_cols and len(numeric_cols) > 0:
                try:
                    date_col = date_cols[0]
                    self.df['temp_date'] = pd.to_datetime(self.df[date_col], errors='coerce')
                    
                    # Select most relevant metric
                    metric_col = numeric_cols[0]
                    
                    plt.figure(figsize=(12, 6))
                    
                    # Prepare data
                    trend_df = self.df.dropna(subset=['temp_date', metric_col])
                    trend_df = trend_df.sort_values('temp_date')
                    
                    if len(trend_df) > 1:
                        # Resample if we have enough data points
                        if len(trend_df) > 20:
                            # Group by appropriate time period
                            time_diff = (trend_df['temp_date'].max() - trend_df['temp_date'].min()).days
                            
                            if time_diff > 365:
                                trend_df = trend_df.set_index('temp_date')
                                trend_data = trend_df[metric_col].resample('M').mean().reset_index()
                                date_format = '%b %Y'
                            elif time_diff > 90:
                                trend_df = trend_df.set_index('temp_date')
                                trend_data = trend_df[metric_col].resample('W').mean().reset_index()
                                date_format = '%d %b'
                            else:
                                trend_data = trend_df
                                date_format = '%d %b'
                                
                            if 'temp_date' not in trend_data.columns:
                                trend_data = trend_data.rename(columns={'index': 'temp_date'})
                        else:
                            trend_data = trend_df
                            date_format = '%d %b'
                            
                        # Create line chart with clear annotations
                        plt.plot(trend_data['temp_date'], trend_data[metric_col], 
                                 marker='o', linestyle='-', linewidth=2, 
                                 color=CORPORATE_COLORS["primary"])
                        
                        # Add trend line
                        z = np.polyfit(range(len(trend_data)), trend_data[metric_col], 1)
                        p = np.poly1d(z)
                        plt.plot(trend_data['temp_date'], p(range(len(trend_data))), 
                                 linestyle='--', color=CORPORATE_COLORS["secondary"], 
                                 alpha=0.8, linewidth=1.5)
                        
                        # Annotations
                        # Mark start and end points
                        plt.plot(trend_data['temp_date'].iloc[0], trend_data[metric_col].iloc[0], 
                                 'o', markersize=8, color=CORPORATE_COLORS["neutral"])
                        plt.plot(trend_data['temp_date'].iloc[-1], trend_data[metric_col].iloc[-1], 
                                 'o', markersize=8, color=CORPORATE_COLORS["accent"])
                        
                        # Add start and end labels
                        plt.annotate(f'{trend_data[metric_col].iloc[0]:.1f}', 
                                     (trend_data['temp_date'].iloc[0], trend_data[metric_col].iloc[0]),
                                     textcoords="offset points", xytext=(-15,-15), 
                                     ha='center', fontweight='bold', fontsize=9)
                        plt.annotate(f'{trend_data[metric_col].iloc[-1]:.1f}', 
                                     (trend_data['temp_date'].iloc[-1], trend_data[metric_col].iloc[-1]),
                                     textcoords="offset points", xytext=(15,-15), 
                                     ha='center', fontweight='bold', fontsize=9)
                        
                        # Calculate change percentage
                        pct_change = ((trend_data[metric_col].iloc[-1] - trend_data[metric_col].iloc[0]) / 
                                      trend_data[metric_col].iloc[0] * 100) if trend_data[metric_col].iloc[0] != 0 else 0
                        
                        # Style and clean up
                        plt.title(f"{metric_col} Trend Analysis", fontsize=14, fontweight='bold')
                        plt.xlabel('')
                        plt.ylabel(metric_col)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        # Format x-axis date labels
                        plt.gcf().autofmt_xdate()
                        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))

                        # Add insight annotation
                        if pct_change > 0:
                            insight_text = f"↑ {pct_change:.1f}% increase"
                            insight_color = CORPORATE_COLORS["success"]
                        else:
                            insight_text = f"↓ {abs(pct_change):.1f}% decrease"
                            insight_color = CORPORATE_COLORS["danger"]
                            
                        plt.figtext(0.5, 0.01, insight_text, ha="center", fontsize=12, 
                                    bbox={"facecolor":insight_color, "alpha":0.2, "pad":5},
                                    fontweight='bold')
                        
                        plt.tight_layout()
                        viz_path = os.path.join(session_dir, "trend_analysis.png")
                        plt.savefig(viz_path, dpi=300, bbox_inches="tight", facecolor='white')
                        plt.close()

                        viz_paths.append({
                            "path": viz_path,
                            "title": f"{metric_col} Trend Over Time",
                            "description": f"Analysis showing a {abs(pct_change):.1f}% {'increase' if pct_change > 0 else 'decrease'} in {metric_col} over the observed time period."
                        })
                
                # Remove temp date column
                self.df.drop('temp_date', axis=1, inplace=True, errors='ignore')
                
            except Exception as e:
                print(f"Error in trend analysis: {str(e)}")
                traceback.print_exc()
            
            # 3. Segment Comparison (if categorical data available)
            categorical_cols = self.df.select_dtypes(include=["object"]).columns
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                segment_col = None
                for col in categorical_cols:
                    n_unique = self.df[col].nunique()
                    if 2 <= n_unique <= 10:  # Find a good segmentation column
                        segment_col = col
                        break
                
                if segment_col:
                    try:
                        plt.figure(figsize=(12, 7))
                        metric_col = numeric_cols[0]
                        
                        # Prepare data
                        segment_data = self.df.groupby(segment_col)[metric_col].agg(['mean', 'count'])
                        segment_data = segment_data.sort_values('mean', ascending=False)
                        
                        # Calculate percentages for annotations
                        total = segment_data['count'].sum()
                        segment_data['pct'] = segment_data['count'] / total * 100
                        
                        # Plot horizontal bar chart
                        bars = plt.barh(segment_data.index, segment_data['mean'], 
                                        color=CORPORATE_COLORS["secondary"], alpha=0.7)
                        
                        # Add bar labels
                        for i, (value, pct) in enumerate(zip(segment_data['mean'], segment_data['pct'])):
                            plt.text(value, i, f" {value:.1f} ({pct:.1f}%)", va='center', fontweight='bold')
                        
                        # Styling
                        plt.title(f"{metric_col} by {segment_col}", fontsize=14, fontweight='bold')
                        plt.xlabel(metric_col)
                        plt.ylabel('')
                        plt.grid(axis='x', linestyle='--', alpha=0.7)
                        plt.grid(axis='y', visible=False)
                        
                        plt.tight_layout()
                        viz_path = os.path.join(session_dir, "segment_comparison.png")
                        plt.savefig(viz_path, dpi=300, bbox_inches="tight", facecolor='white')
                        plt.close()
                        
                        viz_paths.append({
                            "path": viz_path,
                            "title": f"Performance by {segment_col}",
                            "description": f"Comparison of {metric_col} across different {segment_col} segments, highlighting performance variations."
                        })
                    except Exception as e:
                        print(f"Error in segment comparison: {str(e)}")
                        traceback.print_exc()
            
            # 4. Distribution Analysis
            if len(numeric_cols) > 0:
                try:
                    plt.figure(figsize=(10, 6))
                    metric_col = numeric_cols[0]
                    
                    # Create KDE plot with histogram
                    sns.histplot(self.df[metric_col].dropna(), kde=True, color=CORPORATE_COLORS["primary"])
                    
                    # Add mean and median lines
                    mean_val = self.df[metric_col].mean()
                    median_val = self.df[metric_col].median()
                    
                    plt.axvline(mean_val, color=CORPORATE_COLORS["accent"], linestyle='--', linewidth=2)
                    plt.axvline(median_val, color=CORPORATE_COLORS["neutral"], linestyle=':', linewidth=2)
                    
                    # Add legend
                    plt.legend(['Distribution', f'Mean: {mean_val:.2f}', f'Median: {median_val:.2f}'])
                    
                    # Style
                    plt.title(f"Distribution of {metric_col}", fontsize=14, fontweight='bold')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    viz_path = os.path.join(session_dir, "distribution_analysis.png")
                    plt.savefig(viz_path, dpi=300, bbox_inches="tight", facecolor='white')
                    plt.close()
                    
                    viz_paths.append({
                        "path": viz_path,
                        "title": f"Distribution Analysis of {metric_col}",
                        "description": f"Statistical distribution showing the spread and central tendency of {metric_col} values."
                    })
                except Exception as e:
                    print(f"Error in distribution analysis: {str(e)}")
                    traceback.print_exc()
            
            # 5. Correlation Heatmap (if multiple numeric columns)
            if len(numeric_cols) >= 3:
                try:
                    plt.figure(figsize=(10, 8))
                    
                    # Select top metrics
                    selected_cols = numeric_cols[:5]  # Limit to top 5 for readability
                    corr_matrix = self.df[selected_cols].corr()
                    
                    # Create heatmap
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                                linewidths=0.5, cbar_kws={"shrink": 0.8})
                    
                    plt.title("Correlation Between Key Metrics", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    viz_path = os.path.join(session_dir, "correlation_heatmap.png")
                    plt.savefig(viz_path, dpi=300, bbox_inches="tight", facecolor='white')
                    plt.close()
                    
                    viz_paths.append({
                        "path": viz_path,
                        "title": "Metric Correlation Analysis",
                        "description": "Visualization showing relationships between key business metrics, highlighting potential interdependencies."
                    })
                except Exception as e:
                    print(f"Error in correlation analysis: {str(e)}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()
            
        self.visualizations = viz_paths
        return viz_paths

# API Routes and Endpoints would continue here for the FastAPI application
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    business_context: str = Form(None)
):
    session_id = str(uuid.uuid4())
    file_path = f"tmp/uploads/{session_id}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Read data based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")
        
        # Store data in session file for later use
        session_data_path = f"tmp/uploads/{session_id}_data.csv"
        df.to_csv(session_data_path, index=False)
        
        # Run initial analysis
        analyzer = CorporateDataAnalyzer(df, business_context)
        insights = analyzer.generate_business_insights()
        visualizations = analyzer.create_corporate_visualizations(session_id=session_id)
        
        # Return session ID and initial analysis
        return {
            "session_id": session_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
            "insights": insights,
            "recommendations": analyzer.recommendations,
            "visualizations": [{"url": f"/static/visualizations/{session_id}/{os.path.basename(viz['path'])}", 
                              "title": viz["title"], 
                              "description": viz["description"]} 
                             for viz in visualizations]
        }
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/generate-story/")
async def generate_story(request: StoryGenerationRequest):
    """Generate a business data story based on insights and context"""
    try:
        # Validate inputs
        if request.corporate_style not in CORPORATE_STYLES:
            raise HTTPException(status_code=400, detail="Invalid corporate style")
        
        style_info = CORPORATE_STYLES[request.corporate_style]
        
        # Prepare context for LLM
        prompt = f"""
        You are an expert business analyst and data storyteller. Create a concise, professional business report 
        based on the following context:
        
        BUSINESS CONTEXT: {request.business_context}
        
        TARGET AUDIENCE: {request.audience}
        
        STYLE: {style_info['style']}
        
        TONE: {style_info['tone']}
        
        The report should include:
        1. Executive summary (2-3 paragraphs)
        2. Key insights (3-5 bullet points)
        3. Business implications (1-2 paragraphs)
        4. Recommendations (3-5 bullet points)
        5. Next steps (1 paragraph)
        
        Make the content specific, actionable, and valuable to business stakeholders.
        """
        
        # Call LLM (Groq) for generation
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert business analyst and data storyteller."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM API error: {response.text}")
        
        story_content = response.json()["choices"][0]["message"]["content"]
        
        # Create a report ID and title
        report_id = str(uuid.uuid4())
        title = request.title if request.title else f"{style_info['name']} Report"
        
        # Generate PDF
        pdf_path = f"tmp/pdfs/{report_id}.pdf"
        await generate_pdf_report(title, story_content, pdf_path, request.corporate_style)
        
        return StoryResponse(
            report_id=report_id,
            title=title,
            preview=story_content[:300] + "...",
            pdf_url=f"/tmp/pdfs/{report_id}.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

async def generate_pdf_report(title, content, pdf_path, style):
    """Generate a branded PDF report from the story content"""
    try:
        pdf = CorporateReportPDF(title=title)
        
        # Parse content sections (basic parsing)
        sections = {
            "Executive Summary": "",
            "Key Insights": [],
            "Business Implications": "",
            "Recommendations": [],
            "Next Steps": ""
        }
        
        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# ') or line.startswith('## '):
                section_name = line.replace('# ', '').replace('## ', '')
                if section_name in sections:
                    current_section = section_name
            elif current_section:
                if current_section in ["Key Insights", "Recommendations"]:
                    if line.startswith('- ') or line.startswith('* '):
                        sections[current_section].append(line.replace('- ', '').replace('* ', ''))
                else:
                    sections[current_section] += line + "\n"
        
        # Add corporate header based on style
        style_info = CORPORATE_STYLES.get(style, CORPORATE_STYLES["executive"])
        
        # Executive Summary
        pdf.section_title("Executive Summary")
        if sections["Executive Summary"]:
            pdf.section_body(sections["Executive Summary"])
        else:
            pdf.section_body(style_info["intro"])
        
        # Key Insights
        pdf.section_title("Key Insights")
        if sections["Key Insights"]:
            pdf.add_bullet_points(sections["Key Insights"])
        else:
            pdf.add_bullet_points(["Data analysis reveals important patterns that warrant attention", 
                                   "Several metrics show significant deviation from expectations",
                                   "There are clear opportunities for optimization across operations"])
        
        # Business Implications
        pdf.section_title("Business Implications")
        if sections["Business Implications"]:
            pdf.section_body(sections["Business Implications"])
        else:
            pdf.section_body("These findings have substantial implications for business performance and strategy.")
        
        # Recommendations
        pdf.section_title("Recommendations")
        if sections["Recommendations"]:
            pdf.add_bullet_points(sections["Recommendations"])
        else:
            pdf.add_bullet_points(["Implement targeted improvement initiatives based on the data",
                                   "Focus resources on the highest-impact opportunities",
                                   "Monitor progress with regular data review intervals"])
        
        # Next Steps
        pdf.section_title("Next Steps")
        if sections["Next Steps"]:
            pdf.section_body(sections["Next Steps"])
        else:
            pdf.section_body("We recommend a follow-up analysis in 30 days to measure progress against these recommendations.")
        
        # Add footer
        pdf.add_footer()
        
        # Save PDF
        pdf.output(pdf_path)
        return True
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        traceback.print_exc()
        return False

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    pdf_path = f"static/pdfs/{report_id}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(pdf_path)

