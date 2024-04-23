import { useState, useEffect } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { useNavigate, Link } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { register } from "../../actions/userActions";
import Message from "../../components/Message";

const RegisterPage = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { loading, error, userInfo } = useSelector(
    (state) => state.userRegister
  );
  const user = localStorage.getItem("user");

  useEffect(() => {
    if (user) {
      navigate("/");
    }
  }, []);

  useEffect(() => {
    if (userInfo) {
      navigate("/login");
    }
  }, [userInfo]);

  // FORM VALUES
  const [firstName, setFirstName] = useState("Ahsan");
  const [lastName, setLastName] = useState("Umair");
  const [email, setEmail] = useState("ahsanumair@gmail.com");
  const [password, setPassword] = useState("Secret@123");
  const [confirmPassword, setConfirmPassword] = useState("Secret@123");

  const submitHandler = (e) => {
    e.preventDefault();
    dispatch(register(firstName, lastName, email, password, confirmPassword));
  };
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
        {error && <Message variant="danger">{error}</Message>}
        <h1 className="text-center">Sign Up</h1>
        <Row>
          <Col xs={12} lg={6}>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>First Name</Form.Label>
              <Form.Control
                type="text"
                required
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col xs={12} lg={6}>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>Last Name</Form.Label>
              <Form.Control
                type="text"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col xs={12} lg={6}>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>Email Address</Form.Label>
              <Form.Control
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col xs={12} lg={6}>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col xs={12} lg={6}>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>Confirm Password</Form.Label>
              <Form.Control
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </Form.Group>
          </Col>
        </Row>
        <Button type="submit" className="btn-block w-100">
          Register
        </Button>
        <Row className="py-3">
          <Col>
            Have an account? <Link to="/login">Login</Link>
          </Col>
        </Row>
      </Form>
    </Container>
  );
};

export default RegisterPage;
