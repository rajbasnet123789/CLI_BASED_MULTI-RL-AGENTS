const NEXT_STATUS = {
  pending: 'in-progress',
  'in-progress': 'completed',
  completed: 'pending',
};

export default function TaskList({ tasks, projects, onEdit, onDelete, onToggleStatus }) {
  const projectName = (id) => projects.find((project) => project.id === id)?.name || 'Unassigned';

  if (tasks.length === 0) {
    return <p className="empty">No tasks match the current filters.</p>;
  }

  return (
    <ul className="task-list">
      {tasks.map((task) => (
        <li key={task.id} className={`task-card status-${task.status}`}>
          <div className="task-main">
            <div className="task-title-row">
              <h3>{task.title}</h3>
              <span className={`badge ${task.status}`}>{task.status}</span>
            </div>
            {task.description && <p className="task-description">{task.description}</p>}
            <p className="task-meta">Project: {projectName(task.project_id)}</p>
          </div>
          <div className="task-actions">
            <button type="button" onClick={() => onToggleStatus(task)}>
              → {NEXT_STATUS[task.status] || 'pending'}
            </button>
            <button type="button" onClick={() => onEdit(task)}>
              Edit
            </button>
            <button type="button" className="danger" onClick={() => onDelete(task.id)}>
              Delete
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}
