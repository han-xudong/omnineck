from abaqus import *
from odbAccess import *
from abaqusConstants import *
import os
import csv

# Define the finger type
finger_type = "finger_surf"

# Check if the odb path exists
odb_path = os.path.join(".", "data", finger_type, "sim", "abq_files")
if not os.path.exists(odb_path):
    raise FileNotFoundError("Path of odbs does not exist. Please check the finger type or path.")

# Create the csv path if it does not exist
csv_path = os.path.join(".", "data", finger_type, "sim", "csv_files")
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

# Read the motion data from the motion.csv file
with open(os.path.join(".", "data", finger_type, "sim", "motion.csv"), 'r') as f:
        motions = list(csv.reader(f))

# Read odb names from the odbs.csv file
with open(os.path.join(".", "data", finger_type, "sim", "odbs.csv"), 'r') as f:
    odb_names = [line.strip() for line in f.readlines()]

# Read the odb files and extract data
for odb_name in odb_names:
    # Check if the odb file exists
    odb_file = os.path.join(odb_path, odb_name)
    if not os.path.exists(odb_file):
        print("Missing file: {}".format(odb_file))
        continue
    
    # Open the odb file
    odb = openOdb(odb_file)
    
    try:
        # Get U field output
        step = odb.steps['Step-1']
        last_frame = step.frames[-1]
        field = last_frame.fieldOutputs['U']
        
        # Get node values for surface and marker nodes
        surface_node_values = field.getSubset(region=odb.rootAssembly.nodeSets['SET-SURFACE']).values
        marker_node_values = field.getSubset(region=odb.rootAssembly.nodeSets['SET-MARKER']).values
        
        # Get forces on I-SECTION-1
        region = step.historyRegions['Surface SURF-BOTTOM']
        sof1 = region.historyOutputs['SOF1  on section I-SECTION-1'].data[-1][1]
        sof2 = region.historyOutputs['SOF2  on section I-SECTION-1'].data[-1][1]
        sof3 = region.historyOutputs['SOF3  on section I-SECTION-1'].data[-1][1]
        som1 = region.historyOutputs['SOM1  on section I-SECTION-1'].data[-1][1]
        som2 = region.historyOutputs['SOM2  on section I-SECTION-1'].data[-1][1]
        som3 = region.historyOutputs['SOM3  on section I-SECTION-1'].data[-1][1]
    except Exception as e:
        print("Error processing odb file {}: {}".format(odb_name, e))
        odb.close()
        continue
    
    # Get corresponding motion
    index = int(odb_name.replace(".odb", "")) - 1
    motion = motions[index][1:] if index < len(motions) else []
    
    # Save data
    out_file = os.path.join(csv_path, odb_name.replace(".odb", "") + ".csv")
    try:
        with open(out_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["u1", "u2", "u3", "ur1", "ur2", "ur3"])
            writer.writerow(motion)
            writer.writerow(["SOF1", "SOF2", "SOF3", "SOM1", "SOM2", "SOM3"])
            writer.writerow([sof1, sof2, sof3, som1, som2, som3])
            writer.writerow(["surface_node_label", "u1", "u2", "u3"])
            for val in surface_node_values:
                writer.writerow([val.nodeLabel, val.data[0], val.data[1], val.data[2]])
            writer.writerow(["marker_node_label", "u1", "u2", "u3"])
            for val in marker_node_values:
                writer.writerow([val.nodeLabel, val.data[0], val.data[1], val.data[2]])
    except Exception as e:
        print("Error writing to file {}: {}".format(out_file, e))
        continue
    
    # Close the odb file
    odb.close()
    
    # Print completion message
    print("{} done".format(odb_name))