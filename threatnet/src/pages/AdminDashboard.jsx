import React, { useEffect, useState, useRef } from "react";
import DashboardLayout from "../components/DashboardLayout";
import { useAuth } from "../auth/AuthContext";
import { Card, Row, Col, Table, Badge, Button, Form, Spinner, ProgressBar, Alert } from "react-bootstrap";
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const STATUS_COLORS = {
  benign: "#28a745",
  suspicious: "#ffc107",
  critical: "#dc3545",
};

export default function AdminDashboard() {
  const { authAxios } = useAuth(); // Remove user and loadingAuth - ProtectedRoute handles auth
  const [users, setUsers] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [modelMetrics, setModelMetrics] = useState({
    val_accuracy: null,
    trained_examples: 0,
    new_db_examples: 0,
    training: false,
  });
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [loadingAudit, setLoadingAudit] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // Classified messages state
  const [messages, setMessages] = useState([]);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [editId, setEditId] = useState(null);
  const [editText, setEditText] = useState("");
  const [editClassification, setEditClassification] = useState("");

  const pollingRef = useRef(null);

  // 🔹 Fetch data on mount - no auth checks needed, ProtectedRoute handles it
  useEffect(() => {
    fetchUsers();
    fetchAudit();
    fetchMessages();
    fetchModelMetrics();
    // eslint-disable-next-line
  }, []);

  useEffect(() => {
    if (modelMetrics.training || retraining) {
      pollingRef.current = setInterval(fetchModelMetrics, 2000);
    } else {
      clearInterval(pollingRef.current);
    }
    return () => clearInterval(pollingRef.current);
    // eslint-disable-next-line
  }, [modelMetrics.training, retraining]);

  const fetchUsers = async () => {
    setLoadingUsers(true);
    try {
      const res = await authAxios.get("/api/admin/users");
      setUsers(res.data.users || []);
    } catch (e) {
      setUsers([]);
    } finally {
      setLoadingUsers(false);
    }
  };

  const fetchAudit = async () => {
    setLoadingAudit(true);
    try {
      const res = await authAxios.get("/api/audit-logs?limit=20");
      setAuditLogs(res.data.logs || []);
    } catch (e) {
      setAuditLogs([]);
    } finally {
      setLoadingAudit(false);
    }
  };

  const fetchMessages = async () => {
    setLoadingMessages(true);
    try {
      const res = await authAxios.get("/api/admin/messages");
      setMessages(res.data.messages || []);
    } catch (e) {
      setMessages([]);
    } finally {
      setLoadingMessages(false);
    }
  };

  const fetchModelMetrics = async () => {
    try {
      const res = await authAxios.get(`${import.meta.env.VITE_FASTAPI_BASE || "http://localhost:8000"}/api/model/metrics`);
      setModelMetrics((m) => ({
        ...m,
        val_accuracy: res.data.val_accuracy,
        trained_examples: res.data.trained_examples,
        new_db_examples: res.data.new_db_examples,
        training: res.data.training,
      }));
      if (!res.data.training) setRetraining(false);
    } catch (e) {
      // ignore errors if backend not ready
    }
  };

  const triggerRetrain = async () => {
    setRetraining(true);
    setErrorMsg("");
    setModelMetrics((m) => ({ ...m, training: true }));
    try {
      await authAxios.post(`${import.meta.env.VITE_FASTAPI_BASE || "http://localhost:8000"}/api/model/retrain`);
    } catch (e) {
      setModelMetrics((m) => ({ ...m, training: false }));
      setRetraining(false);
      if (e.response && (e.response.status === 401 || e.response.status === 403)) {
        setErrorMsg("You are not authorized to retrain the model.");
      } else {
        setErrorMsg("Failed to start retraining. Please try again or check server logs.");
      }
    }
  };

  const changeRole = async (userId, newRole) => {
    try {
      await authAxios.patch(`/api/admin/users/${userId}/role`, { role: newRole });
      setUsers((u) => u.map((x) => (x.id === userId ? { ...x, role: newRole } : x)));
    } catch (err) {}
  };

  const startEdit = (msg) => {
    setEditId(msg.id);
    setEditText(msg.text);
    setEditClassification(msg.classification);
  };

  const cancelEdit = () => {
    setEditId(null);
    setEditText("");
    setEditClassification("");
  };

  const saveEdit = async (id) => {
    try {
      await authAxios.patch(`/api/admin/messages/${id}`, {
        text: editText,
        classification: editClassification,
      });
      fetchMessages();
      cancelEdit();
    } catch (e) {
      alert("Failed to update message");
    }
  };

  const deleteMessage = async (id) => {
    if (!window.confirm("Delete this message?")) return;
    try {
      await authAxios.delete(`/api/admin/messages/${id}`);
      setMessages((msgs) => msgs.filter((m) => m.id !== id));
    } catch (e) {
      alert("Failed to delete message");
    }
  };

  // Chart data
  const summary = messages.reduce(
    (acc, m) => {
      acc[m.classification] = (acc[m.classification] || 0) + 1;
      return acc;
    },
    { benign: 0, suspicious: 0, critical: 0 }
  );
  const pieData = [
    { name: "Benign", value: summary.benign },
    { name: "Suspicious", value: summary.suspicious },
    { name: "Critical", value: summary.critical },
  ];

  const trendMap = {};
  messages.forEach((m) => {
    const date = m.created_at?.slice(0, 10) || "unknown";
    if (!trendMap[date])
      trendMap[date] = { name: date, benign: 0, suspicious: 0, critical: 0 };
    trendMap[date][m.classification] += 1;
  });
  const lineData = Object.values(trendMap).sort((a, b) =>
    a.name.localeCompare(b.name)
  );

  return (
    <DashboardLayout title="Admin Control Center">
      <Row className="mb-4">
        <Col md={6}>
          <Card className="mb-3 shadow">
            <Card.Header className="bg-gradient text-white" style={{ background: "linear-gradient(90deg, #007bff 0%, #00c6ff 100%)" }}>
              <div className="d-flex align-items-center">
                <span style={{ fontSize: "1.3rem", fontWeight: "bold" }}>
                  <i className="bi bi-bar-chart-line" style={{ marginRight: 8 }}></i>
                  Model Performance
                </span>
                <span className="ms-auto">
                  <Button
                    variant="light"
                    onClick={triggerRetrain}
                    disabled={retraining || modelMetrics.training}
                    size="sm"
                    style={{ fontWeight: "bold" }}
                  >
                    {retraining || modelMetrics.training ? (
                      <>
                        <Spinner animation="border" size="sm" /> Training...
                      </>
                    ) : (
                      <>
                        <i className="bi bi-arrow-repeat"></i> Retrain Model
                      </>
                    )}
                  </Button>
                </span>
              </div>
            </Card.Header>
            <Card.Body>
              {errorMsg && (
                <Alert variant="danger" className="py-2">{errorMsg}</Alert>
              )}
              {modelMetrics.training || retraining ? (
                <div className="text-center my-3">
                  <Spinner animation="border" variant="primary" />
                  <div className="mt-2" style={{ fontWeight: "bold", fontSize: "1.1rem" }}>
                    Model is training...
                  </div>
                  <ProgressBar animated now={100} style={{ height: "8px", marginTop: "10px" }} />
                  <div className="mt-2 text-muted">
                    Please wait until training completes.
                  </div>
                </div>
              ) : (
                <Row>
                  <Col xs={6} className="text-center">
                    <div style={{ marginBottom: 12 }}>
                      <span style={{ fontSize: "2.2rem", fontWeight: "bold", color: "#007bff" }}>
                        {modelMetrics.val_accuracy !== null
                          ? (modelMetrics.val_accuracy * 100).toFixed(2) + "%"
                          : "N/A"}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted" style={{ fontSize: "1rem" }}>
                        <i className="bi bi-check-circle-fill text-info"></i> Validation Accuracy
                      </span>
                    </div>
                  </Col>
                  <Col xs={6}>
                    <div className="mb-3">
                      <Badge bg="info" className="me-2" style={{ fontSize: "1rem" }}>
                        <i className="bi bi-database-fill"></i> Trained Examples
                      </Badge>
                      <span style={{ fontWeight: "bold" }}>
                        {modelMetrics.trained_examples}
                      </span>
                    </div>
                    <div className="mb-3">
                      <Badge bg="success" className="me-2" style={{ fontSize: "1rem" }}>
                        <i className="bi bi-plus-circle-fill"></i> New DB Examples
                      </Badge>
                      <span style={{ fontWeight: "bold" }}>
                        {modelMetrics.new_db_examples}
                      </span>
                    </div>
                  </Col>
                </Row>
              )}
              <hr />
              <div className="text-center">
                <small className="text-muted">
                  <i className="bi bi-lightbulb"></i> Use analyst feedback to improve the model periodically.
                </small>
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card className="mb-3">
            <Card.Header>User Management</Card.Header>
            <Card.Body>
              {loadingUsers ? (
                <div className="text-center">Loading users...</div>
              ) : (
                <Table hover size="sm">
                  <thead>
                    <tr>
                      <th>Username</th>
                      <th>Role</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((u) => (
                      <tr key={u.id}>
                        <td>{u.username}</td>
                        <td>{u.role}</td>
                        <td>
                          <Form.Select
                            size="sm"
                            value={u.role}
                            onChange={(e) => changeRole(u.id, e.target.value)}
                          >
                            <option value="user">user</option>
                            <option value="commander">commander</option>
                            <option value="admin">admin</option>
                          </Form.Select>
                        </td>
                      </tr>
                    ))}
                    {users.length === 0 && (
                      <tr>
                        <td colSpan={3}>No users found</td>
                      </tr>
                    )}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Card className="mb-4">
        <Card.Header>All Classified Messages</Card.Header>
        <Card.Body>
          {loadingMessages ? (
            <div>Loading messages...</div>
          ) : (
            <Table hover responsive>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>User</th>
                  <th>Text</th>
                  <th>Classification</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {messages.length ? (
                  messages.map((m) =>
                    editId === m.id ? (
                      <tr key={m.id}>
                        <td>{m.created_at?.slice(0, 10)}</td>
                        <td>{m.username || "unknown"}</td>
                        <td>
                          <Form.Control
                            as="textarea"
                            rows={2}
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                          />
                        </td>
                        <td>
                          <Form.Select
                            value={editClassification}
                            onChange={(e) => setEditClassification(e.target.value)}
                          >
                            <option value="benign">benign</option>
                            <option value="suspicious">suspicious</option>
                            <option value="critical">critical</option>
                          </Form.Select>
                        </td>
                        <td>
                          <Button size="sm" variant="success" onClick={() => saveEdit(m.id)}>
                            Save
                          </Button>{" "}
                          <Button size="sm" variant="secondary" onClick={cancelEdit}>
                            Cancel
                          </Button>
                        </td>
                      </tr>
                    ) : (
                      <tr key={m.id}>
                        <td>{m.created_at?.slice(0, 10)}</td>
                        <td>{m.username || "unknown"}</td>
                        <td>{m.text}</td>
                        <td>
                          <Badge
                            bg={
                              m.classification === "critical"
                                ? "danger"
                                : m.classification === "suspicious"
                                ? "warning"
                                : "success"
                            }
                          >
                            {m.classification}
                          </Badge>
                        </td>
                        <td>
                          <Button size="sm" variant="primary" onClick={() => startEdit(m)}>
                            Edit
                          </Button>{" "}
                          <Button size="sm" variant="danger" onClick={() => deleteMessage(m.id)}>
                            Delete
                          </Button>
                        </td>
                      </tr>
                    )
                  )
                ) : (
                  <tr>
                    <td colSpan={6} className="text-center">
                      No messages found.
                    </td>
                  </tr>
                )}
              </tbody>
            </Table>
          )}
        </Card.Body>
      </Card>

      <Row className="mb-4">
        <Col md={6}>
          <Card className="mb-3">
            <Card.Header>Classification Distribution</Card.Header>
            <Card.Body style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={50}
                    outerRadius={80}
                    label
                  >
                    {pieData.map((entry, idx) => (
                      <Cell key={idx} fill={STATUS_COLORS[entry.name.toLowerCase()]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        <Col md={6}>
          <Card className="mb-3">
            <Card.Header>Classification Trend</Card.Header>
            <Card.Body style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lineData}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  {Object.entries(STATUS_COLORS).map(([key, color]) => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke={color}
                      strokeWidth={2}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Card className="mb-4">
        <Card.Header>Audit Logs</Card.Header>
        <Card.Body>
          {loadingAudit ? (
            <div>Loading audit logs...</div>
          ) : (
            <Table hover size="sm">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>User</th>
                  <th>Action</th>
                  <th>Success</th>
                </tr>
              </thead>
              <tbody>
                {auditLogs.map((l) => (
                  <tr key={l.id}>
                    <td>{l.created_at}</td>
                    <td>{l.username || "SYSTEM"}</td>
                    <td>{l.action}</td>
                    <td>
                      {l.success ? (
                        <Badge bg="success">✓</Badge>
                      ) : (
                        <Badge bg="danger">✗</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
          <div className="text-end">
            <Button size="sm" onClick={fetchAudit}>
              Refresh
            </Button>
          </div>
        </Card.Body>
      </Card>
    </DashboardLayout>
  );
}
