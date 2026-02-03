"""
Fix emoji-less diagrams for ROS2 graph and Roadmap
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = "docs/images"

C = {
    'bg': '#0d1117', 'card': '#161b22', 'border': '#30363d',
    'text': '#e6edf3', 'dim': '#8b949e', 'blue': '#58a6ff',
    'green': '#3fb950', 'orange': '#d29922', 'red': '#f85149',
    'purple': '#bc8cff', 'cyan': '#39d353', 'pink': '#f778ba',
}

plt.rcParams.update({
    'figure.facecolor': C['bg'], 'axes.facecolor': C['card'],
    'axes.edgecolor': C['border'], 'text.color': C['text'],
    'axes.labelcolor': C['text'], 'xtick.color': C['dim'],
    'ytick.color': C['dim'], 'grid.color': C['border'],
    'font.family': 'sans-serif', 'font.size': 12,
})

# ============================================================
# ROS2 TOPIC GRAPH (no emoji)
# ============================================================
def draw_ros2_graph():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor(C['bg'])
    
    ax.text(7, 7.6, 'ROS2 Node & Topic Graph', ha='center', fontsize=18,
            fontweight='bold', color=C['text'])
    
    def node_box(x, y, label, color):
        rect = FancyBboxPatch((x - 1.3, y - 0.45), 2.6, 0.9,
                               boxstyle="round,pad=0.12",
                               facecolor=color, alpha=0.15, edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color=color)
    
    def topic_oval(x, y, label, color=C['dim']):
        rect = FancyBboxPatch((x - 1.6, y - 0.3), 3.2, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=C['card'], alpha=0.8, edgecolor=color, linewidth=1.5,
                               linestyle='--')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                color=color, fontfamily='monospace')
    
    def arrow(x1, y1, x2, y2, color=C['dim']):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    
    # Nodes (text labels, no emoji)
    node_box(2, 6.5, 'CAMERA', C['orange'])
    node_box(7, 6.5, 'VISION NODE', C['green'])
    node_box(12, 6.5, 'RViz / GUI', C['dim'])
    
    node_box(3.5, 3.5, 'LLM NODE', C['pink'])
    node_box(9, 3.5, 'BRAIN NODE', C['purple'])
    
    node_box(9, 0.8, 'MOTOR DRIVER', C['red'])
    node_box(3.5, 0.8, 'VOICE INPUT', C['orange'])
    
    # Topics
    topic_oval(4.5, 5.5, '/camera/image_raw', C['orange'])
    topic_oval(10, 5.5, '/tinytpu/annotated_image', C['green'])
    topic_oval(7, 4.5, '/tinytpu/detections', C['green'])
    topic_oval(6.2, 2.5, '/tinytpu/llm_input', C['pink'])
    topic_oval(6.2, 1.6, '/tinytpu/llm_output', C['pink'])
    topic_oval(11, 2.2, '/cmd_vel', C['red'])
    topic_oval(11, 4.5, '/tinytpu/status', C['purple'])
    
    # Connections
    arrow(3.3, 6.3, 3.5, 5.8, C['orange'])
    arrow(5.5, 5.5, 5.7, 6.3, C['orange'])
    arrow(7, 6.0, 7, 4.8, C['green'])
    arrow(7.8, 4.5, 8.2, 3.8, C['green'])
    arrow(8.3, 6.5, 8.8, 5.8, C['green'])
    arrow(11.2, 5.5, 11.5, 6.2, C['green'])
    arrow(10, 3.2, 10.5, 2.5, C['purple'])
    arrow(11, 1.9, 10.5, 1.2, C['red'])
    arrow(10, 3.8, 10.5, 4.5, C['purple'])
    arrow(8, 3.3, 7.0, 2.8, C['purple'])
    arrow(5.5, 1.9, 4.5, 3.1, C['pink'])
    arrow(4.5, 3.2, 5.5, 2.8, C['purple'])
    arrow(7.0, 1.6, 8.0, 3.2, C['pink'])
    arrow(3.5, 1.2, 3.5, 3.0, C['orange'])
    
    # Legend
    ax.text(0.5, 0.3, '[solid] = Node (process)', fontsize=9, color=C['blue'])
    ax.text(4.5, 0.3, '[dashed] = Topic (message bus)', fontsize=9, color=C['dim'])
    ax.text(9.5, 0.3, 'arrow = Data flow', fontsize=9, color=C['dim'])
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/ros2_graph.png', dpi=150, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    print(f"  Saved: {OUT}/ros2_graph.png")
    plt.close()


# ============================================================
# ROADMAP (no emoji)
# ============================================================
def draw_roadmap():
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor(C['bg'])
    
    ax.text(8, 6.6, 'TinyTPU Development Roadmap', ha='center', fontsize=18,
            fontweight='bold', color=C['text'])
    
    phases = [
        {
            'name': 'Phase 1\nCore Engine',
            'status': 'COMPLETE',
            'color': C['green'],
            'items': [
                (True, 'Systolic array RTL'),
                (True, 'Python API'),
                (True, 'GPT-2 (15 tok/s)'),
                (True, 'INT8 quant (75%)'),
                (True, 'ONNX engine (50+)'),
            ]
        },
        {
            'name': 'Phase 2\nPerception',
            'status': 'COMPLETE',
            'color': C['green'],
            'items': [
                (True, 'MobileNetV2 13FPS'),
                (True, 'YOLOv5 3 FPS'),
                (True, 'NMS processing'),
                (True, 'Corr = 1.0000'),
            ]
        },
        {
            'name': 'Phase 3\nRobotics',
            'status': 'COMPLETE',
            'color': C['green'],
            'items': [
                (True, 'ROS2 package'),
                (True, 'Vision+LLM+Brain'),
                (True, 'Follow / Avoid'),
                (True, 'NL commands'),
            ]
        },
        {
            'name': 'Phase 4\nDeployment',
            'status': 'IN PROGRESS',
            'color': C['orange'],
            'items': [
                (False, 'Live camera demo'),
                (False, 'Raspberry Pi test'),
                (False, 'MicroROS Pico'),
                (False, 'PyPI package'),
            ]
        },
        {
            'name': 'Phase 5\nAdvanced',
            'status': 'PLANNED',
            'color': C['dim'],
            'items': [
                (False, 'SLAM integration'),
                (False, 'Voice commands'),
                (False, 'TinyLlama on Pi5'),
                (False, 'FPGA deployment'),
            ]
        },
    ]
    
    col_w = 2.9
    gap = 0.15
    
    for i, phase in enumerate(phases):
        x = i * (col_w + gap) + 0.3
        y = 0.5
        h = 5.3
        
        rect = FancyBboxPatch((x, y), col_w, h, boxstyle="round,pad=0.12",
                               facecolor=phase['color'], alpha=0.06,
                               edgecolor=phase['color'], linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x + col_w/2, y + h - 0.55, phase['name'], ha='center',
                fontsize=10, fontweight='bold', color=phase['color'], linespacing=1.3)
        
        badge_color = phase['color']
        ax.text(x + col_w/2, y + h - 1.25, phase['status'], ha='center',
                fontsize=8, fontweight='bold', color=badge_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=C['card'],
                          edgecolor=badge_color, alpha=0.8))
        
        for j, (done, text) in enumerate(phase['items']):
            item_y = y + h - 1.9 - j * 0.7
            if done:
                marker_color = C['green']
                marker = '[X]'
                text_color = C['green']
            else:
                marker_color = C['dim']
                marker = '[ ]'
                text_color = C['dim']
            
            ax.text(x + 0.25, item_y, marker, fontsize=9, va='center',
                    color=marker_color, fontfamily='monospace', fontweight='bold')
            ax.text(x + 0.85, item_y, text, fontsize=9, color=text_color, va='center')
    
    # Progress arrow
    ax.annotate('', xy=(15.3, 0.25), xytext=(0.5, 0.25),
                arrowprops=dict(arrowstyle='->', color=C['blue'], lw=3))
    ax.text(8, 0.05, 'Development Progress', ha='center', fontsize=10, color=C['blue'])
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/roadmap.png', dpi=150, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    print(f"  Saved: {OUT}/roadmap.png")
    plt.close()


print("Regenerating emoji-free diagrams...")
draw_ros2_graph()
draw_roadmap()

print("\nFinal files:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f)) / 1024
    print(f"  {f}: {size:.0f} KB")
