import { useState, useEffect } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { MarkdownRenderer } from '../ui/MarkdownRenderer';
import { formatRelativeTime } from '../../utils';

export function ProjectNotes() {
  const { activeProject, projectNotes } = useDashboardStore();
  const {
    requestActiveProject,
    requestProjectNotes,
    addProjectNote,
    updateProjectNote,
    deleteProjectNote,
  } = useWebSocket();

  const [isAddingNote, setIsAddingNote] = useState(false);
  const [newNoteContent, setNewNoteContent] = useState('');
  const [editingNoteId, setEditingNoteId] = useState<number | null>(null);
  const [editContent, setEditContent] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Fetch active project and notes on mount
  useEffect(() => {
    requestActiveProject();
  }, [requestActiveProject]);

  // Fetch notes when active project changes
  useEffect(() => {
    if (activeProject) {
      requestProjectNotes();
    }
  }, [activeProject, requestProjectNotes]);

  const handleAddNote = () => {
    if (!newNoteContent.trim()) return;
    setIsSaving(true);
    addProjectNote(newNoteContent.trim());
    setTimeout(() => {
      requestProjectNotes();
      setNewNoteContent('');
      setIsAddingNote(false);
      setIsSaving(false);
    }, 500);
  };

  const handleEditNote = (noteId: number, content: string) => {
    setEditingNoteId(noteId);
    setEditContent(content);
  };

  const handleSaveEdit = () => {
    if (editingNoteId === null || !editContent.trim()) return;
    setIsSaving(true);
    updateProjectNote(editingNoteId, editContent.trim());
    setTimeout(() => {
      requestProjectNotes();
      setEditingNoteId(null);
      setEditContent('');
      setIsSaving(false);
    }, 500);
  };

  const handleCancelEdit = () => {
    setEditingNoteId(null);
    setEditContent('');
  };

  const handleDeleteNote = (noteId: number) => {
    if (!confirm('Delete this note?')) return;
    deleteProjectNote(noteId);
    setTimeout(() => {
      requestProjectNotes();
    }, 500);
  };

  return (
    <div className="p-4">
      {/* Active Project Header */}
      <div className="mb-6">
        {activeProject ? (
          <div className="panel p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">📁</span>
              <h3 className="text-sm font-semibold text-text-primary">
                {activeProject.name}
              </h3>
              <span className="badge badge-green">Active</span>
            </div>
            {activeProject.path && (
              <div className="text-xs text-text-muted font-mono truncate">
                {activeProject.path}
              </div>
            )}
            {activeProject.description && (
              <div className="text-sm text-text-secondary mt-2">
                {activeProject.description}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center text-text-muted py-8">
            <div className="text-4xl mb-4">📁</div>
            <div className="text-lg mb-2">No Active Project</div>
            <div className="text-sm">
              Start Workshop in a project directory to track notes here.
            </div>
          </div>
        )}
      </div>

      {/* Add Note Section */}
      {activeProject && (
        <div className="mb-6">
          {isAddingNote ? (
            <div className="panel p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-text-primary">Add Note</h4>
                <div className="text-xs text-text-muted">Supports Markdown</div>
              </div>
              <textarea
                value={newNoteContent}
                onChange={(e) => setNewNoteContent(e.target.value)}
                className="input w-full h-32 font-mono text-sm resize-y"
                placeholder="Write a note about this project..."
                autoFocus
              />
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => {
                    setIsAddingNote(false);
                    setNewNoteContent('');
                  }}
                  className="btn btn-ghost btn-sm"
                  disabled={isSaving}
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddNote}
                  className="btn btn-primary btn-sm"
                  disabled={isSaving || !newNoteContent.trim()}
                >
                  {isSaving ? 'Saving...' : 'Add Note'}
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setIsAddingNote(true)}
              className="btn btn-ghost w-full border border-dashed border-border hover:border-accent-blue"
            >
              + Add Note
            </button>
          )}
        </div>
      )}

      {/* Notes List */}
      {activeProject && projectNotes.length === 0 && !isAddingNote && (
        <div className="text-center text-text-muted py-8">
          <div className="text-sm">
            No notes yet. Add notes to track important information about this project.
          </div>
        </div>
      )}

      {projectNotes.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wide">
            Notes ({projectNotes.length})
          </h4>
          {projectNotes.map((note) => (
            <div
              key={note.id}
              className="panel p-4 border-l-2 border-l-accent-green"
            >
              {editingNoteId === note.id ? (
                <div className="space-y-3">
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="input w-full h-32 font-mono text-sm resize-y"
                    autoFocus
                  />
                  <div className="flex gap-2 justify-end">
                    <button
                      onClick={handleCancelEdit}
                      className="btn btn-ghost btn-sm"
                      disabled={isSaving}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSaveEdit}
                      className="btn btn-primary btn-sm"
                      disabled={isSaving}
                    >
                      {isSaving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="text-xs text-text-muted">
                      {formatRelativeTime(note.created_at)}
                    </div>
                    <div className="flex gap-1">
                      <button
                        onClick={() => handleEditNote(note.id, note.content)}
                        className="btn btn-ghost btn-sm text-xs"
                        title="Edit"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteNote(note.id)}
                        className="btn btn-ghost btn-sm text-xs text-accent-red"
                        title="Delete"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  <MarkdownRenderer content={note.content} />
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
