from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from source.models import reconstruct_dataframe
from source.config import Config

def save_to_png(data_dict, file_name, selected_model, present_date, predict_date, line_style='solid'):

    output_dir = Path(Config.BASE_PATH) / 'outputs' / 'png'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = reconstruct_dataframe(data_dict)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    if isinstance(predict_date, str):
        try:
            predict_date = pd.to_datetime(predict_date)
        except:
            predict_date = datetime.now()
    
    current_time = df['Date'].iloc[present_date]
    current_settlement = df['settlement'].iloc[present_date]
    final_actual_settlement = df['settlement'].iloc[-1]
    final_predicted_settlement = df['predicted_settlement'].iloc[-1]
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(df['Date'], df['settlement'], 
             label='Actual Settlement', 
             color='#1A5F7A',
             linewidth=2,
             marker='o', 
             markersize=4,
             markerfacecolor='#1A5F7A',
             markeredgecolor='white',
             markeredgewidth=1)
    
    predicted_part = df[df['Date'] >= current_time].copy()
    
    if line_style == 'dashed':
        line_style_param = '--'
        line_color = '#FF6B6B'  
    else:
        line_style_param = '-'
        line_color = '#4CAF50'  
    
    plt.plot(predicted_part['Date'], predicted_part['predicted_settlement'], 
             label='Predicted Settlement', 
             color=line_color,
             linestyle=line_style_param,
             linewidth=2,
             marker='s', 
             markersize=4,
             markerfacecolor=line_color,
             markeredgecolor='white',
             markeredgewidth=1)
    
    plt.axvline(current_time, color='#4A4A4A', linestyle='--', linewidth=1.5, 
                label=f'Current Time')
    
    plt.annotate(f'Current Settlement: {current_settlement:.2f} cm', 
                 xy=(current_time, current_settlement),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=9,
                 color='#4A4A4A',
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    plt.annotate(f'Final Actual: {final_actual_settlement:.2f} cm\nFinal Predicted: {final_predicted_settlement:.2f} cm', 
                 xy=(df['Date'].iloc[-1], final_actual_settlement),
                 xytext=(10, -10),
                 textcoords='offset points',
                 fontsize=9,
                 color='#4A4A4A',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.3),
                 ha='left')
    
    plt.title(f"Settlement Prediction for {file_name}\nModel: {selected_model}", 
              fontsize=14, 
              fontweight='bold', 
              color='#2C3E50')
    plt.xlabel("Date", fontsize=10, color='#2C3E50')
    plt.ylabel("Settlement (cm)", fontsize=10, color='#2C3E50')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    plt.legend(loc='best', frameon=True, edgecolor='#4A4A4A', fancybox=True, shadow=True)
    
    plt.grid(True, linestyle='--', linewidth=0.5, color='#E0E0E0')
    
    plt.tight_layout()
    
    start_date_str = current_time.strftime('%Y%m%d')
    end_date_str = df['Date'].iloc[-1].strftime('%Y%m%d')
    output_file_name = f"Settlement_Prediction_{file_name}_{selected_model}_{start_date_str}_to_{end_date_str}.png"
    output_file_path = output_dir / output_file_name
    
    plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f'Graph saved successfully as {output_file_name}.')
    
    return output_file_path

def save_to_csv(data_dict, file_name, selected_model, predict_date):
    output_dir = Path(Config.BASE_PATH) / 'outputs' / 'csv'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_name = f"Settlement_Prediction_{file_name}_{selected_model}_{predict_date}.csv"
    output_file_path = output_dir / output_file_name

    reconstructed_df = reconstruct_dataframe(data_dict)
    reconstructed_df.to_csv(output_file_path, index=False)
    
    return output_file_name