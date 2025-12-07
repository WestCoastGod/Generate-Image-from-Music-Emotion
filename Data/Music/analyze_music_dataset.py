import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

deam_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\Data\Music\DEAM"
pmemo_dir = r"C:\Users\cxoox\Desktop\AIST4010_Project\AIST4010_Project\Data\Music\PMEmo"

print("="*60)
print("MUSIC DATASET ANALYSIS")
print("="*60)

# Load DEAM Dataset
print("\n1. DEAM DATASET")
deam_csv1 = os.path.join(deam_dir, "static_annotations_averaged_songs_1_2000.csv")
deam_csv2 = os.path.join(deam_dir, "static_annotations_averaged_songs_2000_2058.csv")
deam_df1 = pd.read_csv(deam_csv1)
deam_df2 = pd.read_csv(deam_csv2)
deam_df = pd.concat([deam_df1, deam_df2], ignore_index=True)
deam_df.columns = deam_df.columns.str.strip()

print(f"   Total DEAM songs: {len(deam_df)}")
deam_valences = deam_df["valence_mean"].values
deam_arousals = deam_df["arousal_mean"].values
print(f"   Valence: [{deam_valences.min():.2f}, {deam_valences.max():.2f}], mean={deam_valences.mean():.2f}")
print(f"   Arousal: [{deam_arousals.min():.2f}, {deam_arousals.max():.2f}], mean={deam_arousals.mean():.2f}")

# Load PMEmo Dataset
print("\n2. PMEMO DATASET")
pmemo_csv = os.path.join(pmemo_dir, "static_annotations.csv")
pmemo_df = pd.read_csv(pmemo_csv)
pmemo_df.columns = pmemo_df.columns.str.strip()
print(f"   Total PMEmo songs: {len(pmemo_df)}")

pmemo_valences = pmemo_df["Valence(mean)"].values
pmemo_arousals = pmemo_df["Arousal(mean)"].values
print(f"   Valence: [{pmemo_valences.min():.2f}, {pmemo_valences.max():.2f}], mean={pmemo_valences.mean():.2f}")
print(f"   Arousal: [{pmemo_arousals.min():.2f}, {pmemo_arousals.max():.2f}], mean={pmemo_arousals.mean():.2f}")

# Normalize PMEmo to DEAM scale (1-9)
pmemo_valences_scaled = pmemo_valences * 8 + 1
pmemo_arousals_scaled = pmemo_arousals * 8 + 1
print(f"   PMEmo scaled to [1-9]: Valence=[{pmemo_valences_scaled.min():.2f}, {pmemo_valences_scaled.max():.2f}]")
print(f"   PMEmo scaled to [1-9]: Arousal=[{pmemo_arousals_scaled.min():.2f}, {pmemo_arousals_scaled.max():.2f}]")

# Combined dataset
print("\n3. COMBINED DATASET")
total_songs = len(deam_df) + len(pmemo_df)
print(f"   Total songs: {total_songs}")

all_valences = np.concatenate([deam_valences, pmemo_valences_scaled])
all_arousals = np.concatenate([deam_arousals, pmemo_arousals_scaled])
print(f"   Combined Valence: [{all_valences.min():.2f}, {all_valences.max():.2f}], mean={all_valences.mean():.2f}")
print(f"   Combined Arousal: [{all_arousals.min():.2f}, {all_arousals.max():.2f}], mean={all_arousals.mean():.2f}")

# VA Quadrant Analysis
print("\n4. VA QUADRANT DISTRIBUTION")
v_mid = 5.0
a_mid = 5.0
q1 = sum(1 for v, a in zip(all_valences, all_arousals) if v >= v_mid and a >= a_mid)
q2 = sum(1 for v, a in zip(all_valences, all_arousals) if v < v_mid and a >= a_mid)
q3 = sum(1 for v, a in zip(all_valences, all_arousals) if v < v_mid and a < a_mid)
q4 = sum(1 for v, a in zip(all_valences, all_arousals) if v >= v_mid and a < a_mid)
print(f"   Q1 (High V, High A): {q1} ({q1/total_songs*100:.1f}%)")
print(f"   Q2 (Low V, High A):  {q2} ({q2/total_songs*100:.1f}%)")
print(f"   Q3 (Low V, Low A):   {q3} ({q3/total_songs*100:.1f}%)")
print(f"   Q4 (High V, Low A):  {q4} ({q4/total_songs*100:.1f}%)")

# Create visualizations
print("\n5. GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 10))

# Combined VA scatter
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(deam_valences, deam_arousals, alpha=0.5, s=20, c="blue", label=f"DEAM (n={len(deam_df)})")
ax1.scatter(pmemo_valences_scaled, pmemo_arousals_scaled, alpha=0.5, s=20, c="red", label=f"PMEmo (n={len(pmemo_df)})")
ax1.axhline(5, color="gray", linestyle="--", alpha=0.3)
ax1.axvline(5, color="gray", linestyle="--", alpha=0.3)
ax1.set_xlabel("Valence (1-9)")
ax1.set_ylabel("Arousal (1-9)")
ax1.set_title(f"VA Distribution (n={total_songs})")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Valence histogram
ax2 = plt.subplot(2, 3, 2)
ax2.hist(deam_valences, bins=30, alpha=0.5, color="blue", label="DEAM", edgecolor="black")
ax2.hist(pmemo_valences_scaled, bins=30, alpha=0.5, color="red", label="PMEmo", edgecolor="black")
ax2.axvline(all_valences.mean(), color="green", linestyle="--", label=f"Mean={all_valences.mean():.2f}")
ax2.set_xlabel("Valence")
ax2.set_ylabel("Count")
ax2.set_title("Valence Distribution")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Arousal histogram
ax3 = plt.subplot(2, 3, 3)
ax3.hist(deam_arousals, bins=30, alpha=0.5, color="blue", label="DEAM", edgecolor="black")
ax3.hist(pmemo_arousals_scaled, bins=30, alpha=0.5, color="red", label="PMEmo", edgecolor="black")
ax3.axvline(all_arousals.mean(), color="green", linestyle="--", label=f"Mean={all_arousals.mean():.2f}")
ax3.set_xlabel("Arousal")
ax3.set_ylabel("Count")
ax3.set_title("Arousal Distribution")
ax3.legend()
ax3.grid(True, alpha=0.3)

# DEAM only scatter
ax4 = plt.subplot(2, 3, 4)
scatter4 = ax4.scatter(deam_valences, deam_arousals, alpha=0.5, s=20, c=deam_valences, cmap="RdYlGn")
ax4.set_xlabel("Valence (1-9)")
ax4.set_ylabel("Arousal (1-9)")
ax4.set_title(f"DEAM Dataset (n={len(deam_df)})")
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter4, ax=ax4, label="Valence")

# PMEmo only scatter
ax5 = plt.subplot(2, 3, 5)
scatter5 = ax5.scatter(pmemo_valences_scaled, pmemo_arousals_scaled, alpha=0.5, s=20, c=pmemo_valences_scaled, cmap="RdYlGn")
ax5.set_xlabel("Valence (1-9)")
ax5.set_ylabel("Arousal (1-9)")
ax5.set_title(f"PMEmo Dataset (n={len(pmemo_df)})")
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter5, ax=ax5, label="Valence")

# Quadrant pie chart
ax6 = plt.subplot(2, 3, 6)
quadrants = [q1, q2, q3, q4]
labels = [f"Q1: Happy\n({q1})", f"Q2: Tense\n({q2})", f"Q3: Sad\n({q3})", f"Q4: Calm\n({q4})"]
colors = ["#90EE90", "#FFB6C1", "#87CEEB", "#FFD700"]
ax6.pie(quadrants, labels=labels, colors=colors, autopct="%1.1f%%", startangle=45)
ax6.set_title("Emotion Quadrant Distribution")

plt.tight_layout()

# Save figure
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "music_dataset_analysis.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"   Saved visualization to: {output_path}")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
