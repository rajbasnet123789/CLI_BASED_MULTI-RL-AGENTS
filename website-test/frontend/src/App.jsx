import { useCallback, useEffect, useState } from 'react';
import { projectsApi, tasksApi } from './api';
import ProjectManager from './components/ProjectManager';
import TaskFilter from './components/TaskFilter';
import TaskForm from './components/TaskForm';
import TaskList from './components/TaskList';
import './App.css';

export default function App() {
  const [projects, setProjects] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [status, setStatus] = useState('all');
  const [projectId, setProjectId] = useState('');
  const [editingTask, setEditingTask] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const loadProjects = useCallback(async () => {
    const data = await projectsApi.list();
    setProjects(data);
    return data;
  }, []);

  const loadTasks = useCallback(async () => {
    const data = await tasksApi.list({ projectId, status });
    setTasks(data);
  }, [projectId, status]);

  useEffect(() => {
    Promise.all([loadProjects(), loadTasks()])
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [loadProjects, loadTasks]);

  const handleCreateProject = async (name) => {
    const created = await projectsApi.create(name);
    await loadProjects();
    setProjectId(String(created.id));
  };

  const handleRenameProject = async (id, name) => {
    await projectsApi.update(id, name);
    await loadProjects();
    await loadTasks();
  };

  const handleDeleteProject = async (id) => {
    await projectsApi.remove(id);
    if (projectId === String(id)) setProjectId('');
    await loadProjects();
    await loadTasks();
  };

  const handleSubmitTask = async (payload) => {
    try {
      if (editingTask) {
        await tasksApi.update(editingTask.id, payload);
        setEditingTask(null);
      } else {
        await tasksApi.create(payload);
      }
      await loadTasks();
      setError('');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeleteTask = async (id) => {
    try {
      await tasksApi.remove(id);
      await loadTasks();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleToggleStatus = async (task) => {
    const next = { pending: 'in-progress', 'in-progress': 'completed', completed: 'pending' }[
      task.status
    ];
    try {
      await tasksApi.update(task.id, { ...task, status: next });
      await loadTasks();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Task Manager</h1>
          <p>FastAPI + SQLite backend · React + Vite frontend</p>
        </div>
        <span className="header-stat">
          {tasks.length} {tasks.length === 1 ? 'task' : 'tasks'}
        </span>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="app-main">
        <aside>
          <ProjectManager
            projects={projects}
            selectedId={projectId}
            onSelect={(id) => setProjectId(String(id))}
            onCreate={handleCreateProject}
            onRename={handleRenameProject}
            onDelete={handleDeleteProject}
          />
        </aside>

        <section className="content">
          <TaskForm
            projects={projects}
            editingTask={editingTask}
            onSubmit={handleSubmitTask}
            onCancel={() => setEditingTask(null)}
          />

          <div className="list-header">
            <h2>Tasks</h2>
            <TaskFilter
              status={status}
              onStatusChange={setStatus}
              projects={projects}
              projectId={projectId}
              onProjectChange={setProjectId}
            />
          </div>

          {loading ? <p className="empty">Loading…</p> : (
            <TaskList
              tasks={tasks}
              projects={projects}
              onEdit={setEditingTask}
              onDelete={handleDeleteTask}
              onToggleStatus={handleToggleStatus}
            />
          )}
        </section>
      </main>
    </div>
  );
}
