with open('master.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

tex = tex.replace("{plot1.png}", "{experiment_1_validation.png}")
tex = tex.replace("{plot2.png}", "{experiment_2_plot.png}")
tex = tex.replace("{plot3.png}", "{experiment_3_confusion_matrix.png}")
tex = tex.replace("{plot4.png}", "{experiment_4_array_sweep.png}")

with open('master.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print("Stage 1 plots fixed.")
