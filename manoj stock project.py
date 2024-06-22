import pandas as pd

# Load the data
sip_data = pd.read_csv(r"C:\Users\User\Downloads\CIPLA.BO.csv")
nifty_data = pd.read_csv(r"C:\Users\User\Downloads\^NSEI.csv")

# Convert to dictionaries
dict_data_sip = sip_data.set_index('Date')['Close'].to_dict()
dict_data_nifty = nifty_data.set_index('Date')['Close'].to_dict()

# List of dates
dates_sip_list = list(dict_data_sip.keys())

# Initialize variables
count = value_purchased = value_sold = total_profit = total_loss = 0

# Compute profit and loss
for i in range(1, len(dates_sip_list)):  # start from 1 to avoid index error
    current_date = dates_sip_list[i]
    previous_date = dates_sip_list[i - 1]
    sip_current = dict_data_sip[current_date]
    sip_previous = dict_data_sip[previous_date]
    nifty_current = dict_data_nifty[current_date]

    if sip_current > sip_previous:
        count += 1
        value_purchased += nifty_current
    elif sip_current < sip_previous and count > 0:
        count -= 1
        value_sold += nifty_current

    if value_sold != value_purchased:
        difference = value_sold - value_purchased
        if difference > 0:
            total_profit += difference
        else:
            total_loss += -difference

# Print results
print("Value sold:", value_sold)
print("Total purchased:", value_purchased)
print("Total Profit:", total_profit)
print("Total Loss:", total_loss)
print("manoj Velachen ratio:", total_profit / total_loss if total_loss != 0 else "no loss")
