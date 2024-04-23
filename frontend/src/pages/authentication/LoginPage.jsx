import { useEffect, useState } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { useNavigate, Link } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { login } from "../../actions/userActions";
import Message from "../../components/Message";
import Loader from "../../components/Loader";

const LoginPage = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { loading, error, userInfo } = useSelector((state) => state.userLogin);
  const user = localStorage.getItem("user");

  useEffect(() => {
    if (user) {
      navigate("/");
    }
  }, []);

  // FORM VALUES
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const submitHandler = (e) => {
    e.preventDefault();
    dispatch(login(email, password));
  };

  useEffect(() => {
    if (userInfo) {
      if (userInfo.access) {
        navigate("/");
      }
    }
  }, [userInfo]);

  return (
    <Container
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Form
        className="form-control my-5"
        style={{ width: "80%" }}
        onSubmit={submitHandler}
      >
        <h1 className="text-center">Login</h1>
        {error && <Message variant="danger">{error}</Message>}
        {loading && <Loader />}
        <Row>
          <Col xs={12} lg={12}>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>Email</Form.Label>
              <Form.Control
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col xs={12} lg={12}>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </Form.Group>
          </Col>
        </Row>
        <Button type="submit" className="btn-block w-100">
          Submit
        </Button>
        <Row className="py-3">
          <Col>
            Does not have an account? <Link to="/register">Register</Link>
          </Col>
        </Row>
      </Form>
    </Container>
  );
};

export default LoginPage;
