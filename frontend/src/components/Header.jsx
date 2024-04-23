import { Navbar, Nav, NavDropdown, Container } from "react-bootstrap";
import { Link } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { logout } from "../actions/userActions";

const Header = () => {
  const dispatch = useDispatch();
  const userLogin = useSelector((state) => state.userLogin);
  const { userInfo } = userLogin;

  const logoutHandler = () => {
    dispatch(logout());
  };

  return (
    <Navbar collapseOnSelect expand="lg" className="bg-dark" variant="dark">
      <Container fluid>
        <Link className="navbar-brand" to="/">
          My Logo
        </Link>
        <Navbar.Toggle aria-controls="responsive-navbar-nav" />
        <Navbar.Collapse id="responsive-navbar-nav">
          <Nav className="me-auto">
            <Link className="nav-link" to="/">
              List All Users
            </Link>
            <Link className="nav-link" to="/">
              Create New User
            </Link>
          </Nav>
          <Nav>
            <Nav.Link href="#">ahsanumair@gmail.com</Nav.Link>
          </Nav>
          <Nav>
            <Nav.Link href="/login" onClick={logoutHandler}>
              Logout
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default Header;
