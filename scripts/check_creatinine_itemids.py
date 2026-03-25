import pandas as pd
import sys

# Creatinine itemids from d_labitems.csv (blood/serum only)
creat_ids = [50912, 52546, 52024, 51081]

print("Counting creatinine measurements in labevents.csv...")
print(f"Target itemids: {creat_ids}")
print("Processing in chunks (this may take a few minutes)...\n")

chunk_size = 1_000_000
counts = {itemid: 0 for itemid in creat_ids}
total_rows = 0
chunk_num = 0

try:
    for chunk in pd.read_csv('raw_data/labevents.csv', chunksize=chunk_size):
        chunk_num += 1
        total_rows += len(chunk)
        
        # Count occurrences of each creatinine itemid
        for itemid in creat_ids:
            counts[itemid] += (chunk['itemid'] == itemid).sum()
        
        # Progress update every 5 million rows
        if chunk_num % 5 == 0:
            print(f"Processed {total_rows:,} rows...")
            sys.stdout.flush()
    
    print(f"\nTotal rows processed: {total_rows:,}")
    print("\nCreatinine itemid counts:")
    for itemid, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  itemid {itemid}: {count:,} measurements")
    
    total_creat = sum(counts.values())
    print(f"\nTotal creatinine measurements: {total_creat:,}")
    
    # Show which itemids are actually used
    used_itemids = [itemid for itemid, count in counts.items() if count > 0]
    print(f"\nItemids with measurements: {used_itemids}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
