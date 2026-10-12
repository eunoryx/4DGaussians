import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class AdaptiveDensityManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "adaptive_density_viz")
        os.makedirs(self.viz_dir, exist_ok=True)
        self.stats_history = []
        print(f"[Adaptive Density] Initialized. Visualizations will be saved to: {self.viz_dir}")
    
    def adaptive_densify_and_prune(self, gaussians, opt, scene_extent, iteration, stage, densify_threshold, opacity_threshold):
        num_gaussians = gaussians.get_xyz.shape[0]
        
        # Read gradients from gaussians' own accumulators (already populated by add_densification_stats)
        denom = gaussians.denom[:num_gaussians].squeeze()
        denom = torch.clamp(denom, min=1)
        grads = gaussians.xyz_gradient_accum[:num_gaussians].squeeze() / denom
        
        # Use gradient magnitude as complexity score
        complexity_scores = grads.clone()
        if complexity_scores.max() > 0:
            complexity_scores = complexity_scores / complexity_scores.max()
        
        # Define regions
        valid_scores = complexity_scores[complexity_scores > 0]
        if len(valid_scores) > 0:
            high_thresh = torch.quantile(valid_scores, 0.75)
            low_thresh = torch.quantile(valid_scores, 0.25)
        else:
            high_thresh = 0.5
            low_thresh = 0.2
        
        high_complexity = complexity_scores > high_thresh
        low_complexity = complexity_scores < low_thresh
        
        # Adaptive thresholds
        grad_thresh_high = densify_threshold * 0.7
        grad_thresh_normal = densify_threshold
        
        # Densification mask
        selected_pts_mask = torch.zeros(num_gaussians, dtype=torch.bool, device='cuda')
        selected_pts_mask |= (high_complexity & (grads >= grad_thresh_high))
        selected_pts_mask |= (~high_complexity & ~low_complexity & (grads >= grad_thresh_normal))
        
        # Pruning mask: opacity + large low-complexity
        prune_mask = (gaussians.get_opacity[:num_gaussians] < opacity_threshold).squeeze()
        if iteration > 1000:
            scales = gaussians.get_scaling[:num_gaussians].max(dim=1).values
            large_and_simple = low_complexity & (scales > 0.1 * scene_extent)
            prune_mask = prune_mask | large_and_simple
        
        stats = {
            'iteration': iteration,
            'num_gaussians': num_gaussians,
            'high_complexity_count': high_complexity.sum().item(),
            'low_complexity_count': low_complexity.sum().item(),
            'densify_count': selected_pts_mask.sum().item(),
            'prune_count': prune_mask.sum().item(),
            'complexity_scores': complexity_scores.cpu(),
            'high_complexity': high_complexity.cpu(),
            'low_complexity': low_complexity.cpu()
        }
        self.stats_history.append(stats)
        # Positions the scores above belong to; the model is pruned and densified before visualize() runs
        self.last_xyz = gaussians.get_xyz.detach().cpu().numpy()

        print(f"\n[Adaptive Density - Iter {iteration}]")
        print(f"  Total: {num_gaussians} | High: {stats['high_complexity_count']} | Low: {stats['low_complexity_count']}")
        print(f"  To Densify: {stats['densify_count']} | To Prune: {stats['prune_count']}")
        
        return selected_pts_mask, prune_mask, stats
    
    def visualize(self, gaussians, iteration):
        if len(self.stats_history) == 0:
            return
        
        stats = self.stats_history[-1]
        xyz = self.last_xyz
        scores = stats['complexity_scores'].numpy()
        high_comp = stats['high_complexity'].numpy()
        low_comp = stats['low_complexity'].numpy()
        
        min_size = min(len(xyz), len(scores))
        xyz = xyz[:min_size]
        scores = scores[:min_size]
        high_comp = high_comp[:min_size]
        low_comp = low_comp[:min_size]
        
        max_points = 10000
        if len(xyz) > max_points:
            indices = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[indices]
            scores = scores[indices]
            high_comp = high_comp[indices]
            low_comp = low_comp[indices]
        
        fig = plt.figure(figsize=(20, 6))
        
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                              c=scores, cmap='hot', s=1, alpha=0.5)
        ax1.set_title(f'Complexity Distribution\n(Iteration {iteration})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.8)
        cbar1.set_label('Complexity', rotation=270, labelpad=15)
        
        ax2 = fig.add_subplot(132, projection='3d')
        colors = np.zeros(len(xyz))
        colors[high_comp] = 2
        colors[low_comp] = 1
        cmap = plt.cm.colors.ListedColormap(['green', 'blue', 'red'])
        scatter2 = ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                              c=colors, cmap=cmap, s=1, alpha=0.5)
        ax2.set_title('Adaptive Regions', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor='red', label=f'High ({high_comp.sum()} pts)'),
            Patch(facecolor='green', label=f'Med ({(~high_comp & ~low_comp).sum()} pts)'),
            Patch(facecolor='blue', label=f'Low ({low_comp.sum()} pts)')
        ]
        ax2.legend(handles=legend, loc='upper left', fontsize=9)
        
        ax3 = fig.add_subplot(133)
        ax3.hist(scores[scores > 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        if high_comp.sum() > 0:
            ax3.axvline(scores[high_comp].min(), color='red', linestyle='--', linewidth=2, label='High thresh')
        if low_comp.sum() > 0:
            ax3.axvline(scores[low_comp].max(), color='blue', linestyle='--', linewidth=2, label='Low thresh')
        ax3.set_xlabel('Complexity Score', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, f'complexity_iter_{iteration:05d}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved: {save_path}")
    
    def create_summary(self):
        if len(self.stats_history) < 2:
            return
        
        iterations = [s['iteration'] for s in self.stats_history]
        total_counts = [s['num_gaussians'] for s in self.stats_history]
        high_counts = [s['high_complexity_count'] for s in self.stats_history]
        low_counts = [s['low_complexity_count'] for s in self.stats_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(iterations, total_counts, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Iteration', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title('Total Gaussian Count', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(iterations, high_counts, 'r-', linewidth=2, marker='s', markersize=4, label='High')
        axes[0, 1].plot(iterations, low_counts, 'b-', linewidth=2, marker='^', markersize=4, label='Low')
        axes[0, 1].set_xlabel('Iteration', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('Complexity Regions', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        med_counts = np.array(total_counts) - np.array(high_counts) - np.array(low_counts)
        axes[1, 0].fill_between(iterations, 0, high_counts, alpha=0.6, color='red', label='High')
        axes[1, 0].fill_between(iterations, high_counts, np.array(high_counts) + med_counts, 
                               alpha=0.6, color='green', label='Medium')
        axes[1, 0].fill_between(iterations, np.array(high_counts) + med_counts, total_counts, 
                               alpha=0.6, color='blue', label='Low')
        axes[1, 0].set_xlabel('Iteration', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Stacked Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        high_pct = np.array(high_counts) / np.array(total_counts) * 100
        low_pct = np.array(low_counts) / np.array(total_counts) * 100
        axes[1, 1].plot(iterations, high_pct, 'r-', linewidth=2, marker='s', markersize=4, label='High %')
        axes[1, 1].plot(iterations, low_pct, 'b-', linewidth=2, marker='^', markersize=4, label='Low %')
        axes[1, 1].set_xlabel('Iteration', fontsize=11)
        axes[1, 1].set_ylabel('Percentage (%)', fontsize=11)
        axes[1, 1].set_title('Region Percentages', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'adaptive_density_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[Adaptive Density] Summary saved: {save_path}")
    
    def reset(self):
        """No-op: gradients are handled by GaussianModel"""
        pass