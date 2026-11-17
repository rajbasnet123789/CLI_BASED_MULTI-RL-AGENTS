import { useEffect, useState } from 'react';

const EMPTY = { title: '', description: '', status: 'pending', project_id: '' };

export default function TaskForm({ projects, editingTask, onSubmit, onCancel }) {
  const [form, setForm] = useState(EMPTY);

  useEffect(() => {
    if (editingTask) {
      setForm({
        title: editingTask.title,
        description: editingTask.description || '',
        status: editingTask.status,
        project_id: String(editingTask.project_id),
      });
    } else {
      setForm(EMPTY);
    }
  }, [editingTask]);

  const setField = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!form.title.trim() || !form.project_id) return;
    onSubmit({
      title: form.title.trim(),
      description: form.description.trim() || null,
      status: form.status,
      project_id: Number(form.project_id),
    });
    setForm(EMPTY);
  };

  return (
    <form className="task-form" onSubmit={handleSubmit}>
      <h2>{editingTask ? 'Edit task' : 'New task'}</h2>
      <div className="form-grid">
        <label>
          Title
          <input
            value={form.title}
            placeholder="What needs doing?"
            onChange={(event) => setField('title', event.target.value)}
            required
          />
        </label>
        <label>
          Project
          <select
            value={form.project_id}
            onChange={(event) => setField('project_id', event.target.value)}
            required
          >
            <option value="">Select a project</option>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Status
          <select value={form.status} onChange={(event) => setField('status', event.target.value)}>
            <option value="pending">Pending</option>
            <option value="in-progress">In progress</option>
            <option value="completed">Completed</option>
          </select>
        </label>
        <label className="full-width">
          Description
          <textarea
            value={form.description}
            placeholder="Optional details"
            rows="2"
            onChange={(event) => setField('description', event.target.value)}
          />
        </label>
      </div>
      <div className="form-actions">
        <button type="submit" className="primary">
          {editingTask ? 'Save changes' : 'Add task'}
        </button>
        {editingTask && (
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
