from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_weekly_forecast(forecast_csv_path):
    """Generate separate PNG files for each week"""
    # Load the predictions
    df = pd.read_csv(forecast_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # weekday() returns 0 for Monday, 6 for Sunday
    df["day_index"] = df["timestamp"].dt.weekday
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    # Map timestamps to a continuous 0.0 to 7.0 X-axis (Mon 00:00 to Sun 23:30)
    df["timeline_position"] = df["day_index"] + (df["hour"] / 24.0)

    # Group by week
    df["week_number"] = ((df["timestamp"] - df["timestamp"].min()).dt.days // 7) + 1
    
    # Limit to first 4 weeks
    weeks_to_plot = min(4, len(df["week_number"].unique()))
    
    vmin, vmax = 0, 24  # Fixed hour range for color mapping
    
    for i in range(weeks_to_plot):
        week_data = df[df["week_number"] == i + 1]
        
        if len(week_data) == 0:
            continue
        
        plt.figure(figsize=(14, 6))
        
        # Create scatter plot with consistent color mapping
        scatter = plt.scatter(
            week_data["timeline_position"],
            week_data["traffic_mb"],
            c=week_data["hour"],
            cmap="plasma",  
            vmin=vmin, 
            vmax=vmax,      
            alpha=0.8,
            s=25,
            edgecolors="none",
            label=f"Week {i+1} Forecast",
        )
        
        # Formatting axes and labels
        days_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        plt.xticks(np.arange(7) + 0.5, days_labels, fontsize=11)
        plt.xlim(0, 7)
        plt.ylim(bottom=0)
        
        # Calculate week date range
        start_date = week_data["timestamp"].min().strftime("%Y-%m-%d")
        end_date = week_data["timestamp"].max().strftime("%Y-%m-%d")
        plt.title(
            f"Network Traffic Profile - Week {i+1}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Day of the Week", fontsize=12)
        plt.ylabel("Traffic (MB)", fontsize=12)
        
        # Add colorbar with consistent scale
        cbar = plt.colorbar(scatter, pad=0.02)
        cbar.set_label("Hour of Day (24h Scale)", fontsize=11)
        cbar.set_ticks([0, 6, 12, 18, 24])
        cbar.set_ticklabels(["00:00 (Night)", "06:00", "12:00 (Noon)", "18:00", "24:00"])
        
        # Clean up background grid lines
        plt.grid(True, linestyle=":", alpha=0.6)
        
        # Save individual week image
        output_image = Path(forecast_csv_path).parent / f"network_traffic_week_{i+1}.png"
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        print(f"Week {i+1} visualization saved as: {output_image}")
        
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Look for the file in the script's local directory
    csv_file = Path(__file__).with_name("network_traffic_data.csv")

    if csv_file.exists():
        visualize_weekly_forecast(csv_file)
    else:
        print(
            f"Error: Could not find '{csv_file.name}'. "
            "Please run your data_generation.py script first to generate the data!"
        )