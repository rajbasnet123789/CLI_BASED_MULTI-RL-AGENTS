const STATUSES = ['all', 'pending', 'in-progress', 'completed'];

export default function TaskFilter({ status, onStatusChange, projects, projectId, onProjectChange }) {
  return (
    <div className="filters">
      <label>
        Status
        <select value={status} onChange={(event) => onStatusChange(event.target.value)}>
          {STATUSES.map((value) => (
            <option key={value} value={value}>
              {value === 'all' ? 'All statuses' : value}
            </option>
          ))}
        </select>
      </label>

      <label>
        Project
        <select value={projectId} onChange={(event) => onProjectChange(event.target.value)}>
          <option value="">All projects</option>
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
