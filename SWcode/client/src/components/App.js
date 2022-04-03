import React, { Suspense } from 'react';
import { Route, Switch } from "react-router-dom";
import Auth from "../hoc/auth";
// pages for this product
import LandingPage from "./views/LandingPage/LandingPage.js";
import HomePage from "./views/LandingPage/HomePage.js";
import LoginPage from "./views/LoginPage/LoginPage.js";
import RegisterPage from "./views/RegisterPage/RegisterPage.js";
import NavBar from "./views/NavBar/NavBar";
import Footer from "./views/Footer/Footer"
import UploadProductPage from "./views/UploadProductPage/UploadProductPage.js";
import DetailProductPage from "./views/DetailProductPage/DetailProductPage";
import DipsPage from './views/DipsPage/DipsPage';
import HistoryPage from './views/HistoryPage/HistoryPage';
import RequestPage from './views/RequestPage/RequestPage';
import UploadRequestPage from "./views/UploadProductPage/UploadRequestPage.js";
import DetailRequestPage from "./views/DetailRequestPage/DetailRequestPage.js";
import AllowProductPage2 from "./views/UploadProductPage/AllowProductPage2";
import AllowWaitingPage from "./views/AllowWaitngPage/AllowWaitingPage";
//null   Anyone Can go inside
//true   only logged in user can go inside
//false  logged in user can't go inside

function App() {
  return (
    <Suspense fallback={(<div>Loading...</div>)}>
      <NavBar />
      <div style={{ paddingTop: '69px', minHeight: 'calc(100vh - 80px)' }}>
        <Switch>
          <Route exact path="/" component={Auth(LandingPage, null)} />
          <Route exact path="/Menu" component={Auth(HomePage, null)} />
          <Route exact path="/login" component={Auth(LoginPage, false)} />
          <Route exact path="/register" component={Auth(RegisterPage, false)} />
          <Route exact path="/product/upload" component={Auth(UploadProductPage, true)} />
          <Route exact path="/request/upload" component={Auth(UploadRequestPage, true)} />
          <Route exact path="/product/:productId" component={Auth(DetailProductPage, null)} />
          <Route exact path="/user/cart" component={Auth(DipsPage, true)} />
          <Route exact path="/history" component={Auth(HistoryPage, true)} />
          <Route exact path="/RequestPage" component={Auth(RequestPage, true)} />
          <Route exact path="/request/:requestId" component={Auth(DetailRequestPage, true)} />
          <Route exact path="/allow/:productId" component={Auth(AllowProductPage2, true)} />
          <Route exact path="/allow" component={Auth(AllowWaitingPage, true)} />
        </Switch>
      </div>
      <Footer />
    </Suspense>
  );
}

export default App;
