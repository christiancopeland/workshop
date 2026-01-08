import { useState, useEffect } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { MarkdownRenderer } from '../ui/MarkdownRenderer';

export function UserProfile() {
  const { memoryProfile } = useDashboardStore();
  const { updateProfile, requestMemoryProfile } = useWebSocket();

  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Sync edit content when profile changes
  useEffect(() => {
    if (!isEditing) {
      setEditContent(memoryProfile);
    }
  }, [memoryProfile, isEditing]);

  const handleEdit = () => {
    setEditContent(memoryProfile);
    setIsEditing(true);
  };

  const handleCancel = () => {
    setEditContent(memoryProfile);
    setIsEditing(false);
  };

  const handleSave = async () => {
    setIsSaving(true);
    updateProfile(editContent);
    // Wait a moment then refresh
    setTimeout(() => {
      requestMemoryProfile();
      setIsSaving(false);
      setIsEditing(false);
    }, 500);
  };

  return (
    <div className="p-4">
      {!memoryProfile && !isEditing ? (
        <div className="text-center text-text-muted py-12">
          <div className="text-4xl mb-4">👤</div>
          <div className="text-lg mb-2">No User Profile Yet</div>
          <div className="text-sm mb-4">
            The agent learns about you over time, or you can create a profile manually.
          </div>
          <button onClick={handleEdit} className="btn btn-primary">
            Create Profile
          </button>
        </div>
      ) : isEditing ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-text-primary">
              Edit Profile
            </h3>
            <div className="text-xs text-text-muted">
              Supports Markdown
            </div>
          </div>

          <textarea
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            className="input w-full h-64 font-mono text-sm resize-y"
            placeholder="Write about yourself, your preferences, work context, etc. The agent will use this to personalize responses.

Example:
# About Me
- Software engineer focused on Python and TypeScript
- Working on AI/ML projects
- Prefer concise, technical explanations

# Preferences
- Code examples over lengthy explanations
- Dark mode enthusiast
- Use metric units"
          />

          <div className="flex gap-2 justify-end">
            <button
              onClick={handleCancel}
              className="btn btn-ghost"
              disabled={isSaving}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="btn btn-primary"
              disabled={isSaving}
            >
              {isSaving ? 'Saving...' : 'Save Profile'}
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-2xl">👤</span>
              <h3 className="text-sm font-semibold text-text-primary">
                User Profile
              </h3>
            </div>
            <button onClick={handleEdit} className="btn btn-ghost btn-sm">
              Edit
            </button>
          </div>

          <div className="panel p-4">
            <MarkdownRenderer content={memoryProfile} />
          </div>
        </div>
      )}
    </div>
  );
}
