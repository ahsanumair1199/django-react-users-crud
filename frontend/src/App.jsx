import React from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { LoginPage, RegisterPage, HomePage } from "./pages";
import ProtectedRoutes from "./ProtectedRoutes";

const router = createBrowserRouter([
  {
    path: "/login",
    element: <LoginPage />,
  },
  {
    path: "register/",
    element: <RegisterPage />,
  },
  {
    element: <ProtectedRoutes />,
    children: [
      {
        path: "/",
        element: <HomePage />,
      },
    ],
  },
]);

const App = () => {
  return <RouterProvider router={router} />;
};

export default App;
