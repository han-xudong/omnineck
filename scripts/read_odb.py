import os
import csv
from abaqus import *
from odbAccess import *
from abaqusConstants import *


# Define the omnineck type
omnineck_type = "omnineck"

# Define the data directory
data_dir = os.path.join(".", "data", omnineck_type, "sim")
if not os.path.exists(data_dir):
    raise FileNotFoundError("Data directory does not exist.")

# Check if the odb path exists
odb_path = os.path.join(data_dir, "abq_files")
if not os.path.exists(odb_path):
    raise FileNotFoundError("Path of odbs does not exist.")

# Create the csv path if it does not exist
csv_path = os.path.join(data_dir, "csv_files")
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

# Read the motion data from the motion.csv file
with open(os.path.join(data_dir, "motion.csv"), "r") as f:
    motions = list(csv.reader(f))

# Read odb names from the odbs.csv file
with open(os.path.join(data_dir, "odbs.csv"), "r") as f:
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
        # Set step
        step = odb.steps["Step-1"]

        # Get the last frame
        last_frame = step.frames[-1]

        # Get the field outputs
        field = last_frame.fieldOutputs["U"]

        # Set regions
        surface_region = odb.rootAssembly.nodeSets["SET-SURFACE"]
        bound_region = odb.rootAssembly.nodeSets["SET-BOUND"]

        # Get node values
        surface_node_values = field.getSubset(region=surface_region).values
        bound_node_values = field.getSubset(region=bound_region).values

        # Get region
        region = step.historyRegions["Surface SURF-BOTTOM"]

        # Extract forces from history outputs
        sof1 = region.historyOutputs["SOF1  on section I-SECTION-1"].data[-1][1]
        sof2 = region.historyOutputs["SOF2  on section I-SECTION-1"].data[-1][1]
        sof3 = region.historyOutputs["SOF3  on section I-SECTION-1"].data[-1][1]
        som1 = region.historyOutputs["SOM1  on section I-SECTION-1"].data[-1][1]
        som2 = region.historyOutputs["SOM2  on section I-SECTION-1"].data[-1][1]
        som3 = region.historyOutputs["SOM3  on section I-SECTION-1"].data[-1][1]
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
        with open(out_file, "wb") as f:
            writer = csv.writer(f)

            # Write motion
            writer.writerow(["u1", "u2", "u3", "ur1", "ur2", "ur3"])
            writer.writerow(motion)

            # Write force
            writer.writerow(["SOF1", "SOF2", "SOF3", "SOM1", "SOM2", "SOM3"])
            writer.writerow([sof1, sof2, sof3, som1, som2, som3])

            # Write surface node values
            writer.writerow(["surface_node_label", "u1", "u2", "u3"])
            for val in surface_node_values:
                writer.writerow([val.nodeLabel, val.data[0], val.data[1], val.data[2]])

            # Write bound node values
            writer.writerow(["bound_node_label", "u1", "u2", "u3"])
            for val in bound_node_values:
                writer.writerow([val.nodeLabel, val.data[0], val.data[1], val.data[2]])
    except Exception as e:
        print("Error writing to file {}: {}".format(out_file, e))
        continue

    # Close the odb file
    odb.close()

    # Print completion message
    print("{} completed".format(odb_name))

print("All odb files processed.")