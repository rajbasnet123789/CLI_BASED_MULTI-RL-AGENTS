import { useState } from 'react';

export default function ProjectManager({ projects, selectedId, onSelect, onCreate, onRename, onDelete }) {
  const [name, setName] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingName, setEditingName] = useState('');
  const [error, setError] = useState('');

  const handleCreate = async (event) => {
    event.preventDefault();
    if (!name.trim()) return;
    try {
      await onCreate(name.trim());
      setName('');
      setError('');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRename = async (id) => {
    if (!editingName.trim()) return;
    try {
      await onRename(id, editingName.trim());
      setEditingId(null);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <section className="panel projects">
      <h2>Projects</h2>

      <form className="inline-form" onSubmit={handleCreate}>
        <input
          aria-label="New project name"
          placeholder="New project name"
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
        <button type="submit">Add</button>
      </form>

      {error && <p className="error">{error}</p>}

      <div className="project-list">
        <button
          type="button"
          className={`project-chip ${selectedId === '' ? 'active' : ''}`}
          onClick={() => onSelect('')}
        >
          All projects
        </button>
        {projects.map((project) => (
          <div key={project.id} className={`project-row ${selectedId === project.id ? 'active' : ''}`}>
            {editingId === project.id ? (
              <form
                className="inline-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  handleRename(project.id);
                }}
              >
                <input
                  aria-label="Rename project"
                  value={editingName}
                  onChange={(event) => setEditingName(event.target.value)}
                  autoFocus
                />
                <button type="submit">Save</button>
              </form>
            ) : (
              <>
                <button type="button" className="project-name" onClick={() => onSelect(project.id)}>
                  {project.name}
                </button>
                <button
                  type="button"
                  className="icon-button"
                  title="Rename"
                  onClick={() => {
                    setEditingId(project.id);
                    setEditingName(project.name);
                  }}
                >
                  ✎
                </button>
                <button
                  type="button"
                  className="icon-button danger"
                  title="Delete"
                  onClick={() => onDelete(project.id)}
                >
                  ×
                </button>
              </>
            )}
          </div>
        ))}
        {projects.length === 0 && <p className="empty">No projects yet. Add one above.</p>}
      </div>
    </section>
  );
}
