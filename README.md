### Uber vs Bolt Nairobi Pricing Analysis - REAL API Data
#### Project Overview

A comprehensive real-time pricing analysis tool that compares Uber and Bolt ride-hailing services across Nairobi using actual API data and sophisticated data visualization.

#### Features
Core Functionality

- Real API Integration: Connects to actual Uber and Bolt APIs for live pricing data

- Nairobi-Specific Analysis: Focused on popular Nairobi locations and routes

- Advanced Data Collection: Automated sampling across multiple routes and time periods

- Comprehensive Analytics: Statistical analysis with confidence intervals and significance testing

- Enhanced Visualization: 12+ detailed charts and interactive dashboards

#### Key Analyses
- Price Comparison: Side-by-side Uber vs Bolt pricing

- Time-based Analysis: Rush hour vs off-peak pricing

- Distance Efficiency: Cost per kilometer analysis

- Surge Pricing Impact: Multiplier effects on both platforms

- Route-specific Insights: Top routes with maximum savings potential

- Statistical Significance: T-tests and confidence intervals

#### Installation & Setup
- Prerequisites
- Python 3.7+

- Required packages (see requirements below)

- RapidAPI account for Uber/Bolt API access

#### Required Packages
-  bash
-  pip install pandas numpy matplotlib seaborn requests python-dotenv scipy
-  Environment Setup
-  Clone the repository

Install required packages

-  Set up your .env file with API credentials:

text
-  RAPIDAPI_KEY=your_rapidapi_key_here
  
#### Project Structure
text
├── main_analysis.py          # Main execution file

├── .env                      # Environment variables (API keys)

├── chart1_price_comparison_horizontal_bars.png

├── chart2_price_difference_kde.png

├── ... (all generated charts)

└── nairobi_enhanced_analysis_[timestamp].csv

##### 🔧 Usage

  Basic Execution
  python
  python main_analysis.py
  Custom Configuration
  
-  Modify n_samples in collect_live_data() for more/fewer data points

-  Adjust location coordinates in get_nairobi_locations()

-  Customize chart parameters in individual chart methods

#### Data Collection

The system collects data from:

-  Real APIs: When available and responsive

-  Enhanced Fallback: Realistic simulated data based on Nairobi market rates

-  Multiple Routes: Random sampling across 16+ Nairobi locations

#### Output & Results

Generated Charts

-  Horizontal Bar Comparison - Comprehensive price metrics

-  KDE Price Distribution - Statistical distribution analysis

-  Cheaper Option Analysis - Pie chart of platform advantage

-  Data Quality Overview - Source distribution

-  Time-based Pricing - Hourly trends with confidence intervals

-  Savings Analysis - Potential savings by time

-  Cost Efficiency - Per-kilometer cost analysis

-  Top Routes - Route-specific price differences

-  Surge Pricing Impact - Multiplier effects

-  Distance-Price Relationship - Scatter plots with trend lines

-  Interactive Dashboard - Real-time recommendations

-  Summary Statistics - Comprehensive overview

Key Metrics
-  Average price differences

-  Statistical significance (p-values)

-  Savings potential calculations

-  Platform preference recommendations

-  Time-based optimization strategies

#### 🎯 Key Insights
Typical Findings
-  Bolt Advantage: Often 5-15% cheaper on average

-  Surge Patterns: Higher premiums during rush hours (7-9 AM, 5-7 PM)

-  Distance Efficiency: Better per-km rates on longer trips

-  Time Optimization: Specific hours with maximum savings potential

Actionable Recommendations
-  Platform preference based on time of day

-  Route-specific optimization strategies

-  Monthly savings potential calculations

-  Real-time booking recommendations

#### 🔒 Data Sources & Quality
Primary Sources
-  Uber API: Real-time price estimates

-  Bolt API: Live pricing data

-  RapidAPI: API gateway services

Data Quality Measures
-  Automatic NaN/inf value handling

-  Outlier detection and removal

-  Confidence interval calculations

-  Sample size validation

Fallback System
-  When APIs are unavailable:

-  Realistic Nairobi-based pricing models

-  Traffic pattern simulations

-  Surge pricing algorithms

Distance-based calculations

#### 📊 Statistical Methods
Analysis Techniques
-  T-tests: Statistical significance of price differences

-  Confidence Intervals: 95% CI for all averages

-  Correlation Analysis: Distance vs price relationships

-  Percentile Analysis: Price distribution percentiles

-  Trend Analysis: Linear regression for price trends

Validation
-  Sample size adequacy checks

-  Data distribution normality tests

-  Outlier impact assessment

Multiple comparison corrections

#### 💡 Business Applications
For Riders
-  Cost optimization strategies

-  Real-time booking decisions

-  Monthly budget planning

-  Route selection guidance

For Researchers
-  Nairobi transportation economics

-  Ride-hailing market dynamics

-  Pricing strategy analysis

-  Urban mobility patterns

#### 🚨 Limitations & Considerations
Current Limitations
-  API rate limits may affect data collection

-  Simulated data used when APIs are unavailable

-  Nairobi-specific (not directly applicable to other cities)

-  Real-time factors may affect accuracy

#### Future Enhancements
-  Additional ride-hailing platforms

-  Historical trend analysis

-  Weather impact integration

-  Traffic condition correlations

#### Support & Contribution
Issues & Questions
-  Check generated error logs

-  Verify API key validity

-  Ensure all dependencies are installed

-  Review data quality outputs


📄 License & Attribution

This project is intended for educational and research purposes. Please ensure compliance with:

-    Uber API Terms of Service

-    Bolt API Terms of Service

-    RapidAPI usage policies

-    Local data protection regulations
