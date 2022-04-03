import React from 'react';
import { Menu } from 'antd';
import { useSelector } from "react-redux";
const SubMenu = Menu.SubMenu;
const MenuItemGroup = Menu.ItemGroup;

function LeftMenu(props) {

  const user = useSelector(state => state.user)


  if (user.userData && !user.userData.isAuth) {
    return (
      <Menu mode={props.mode}>
      <Menu.Item key="mail">
        <a href="/">홈</a>
      </Menu.Item>

      
      
    </Menu>
    )
  } else if(user.userData && user.userData.role == 2) {
  return (
      <Menu mode={props.mode}>
      <Menu.Item key="mail">
        <a href="/">홈</a>
      </Menu.Item>

      {/* <Menu.Item>
          <a href="/RequestPage">문의 게시판</a>
      </Menu.Item> */}
      
      <Menu.Item>
          <a href="/Allow">승인대기</a>
      </Menu.Item>
            
      {/* <Menu.Item>
          <a href="/Menu">메뉴판</a>
      </Menu.Item> */}
    </Menu>
    )
  } else {
    return (
      <Menu mode={props.mode}>
      <Menu.Item key="mail">
        <a href="/">홈</a>
      </Menu.Item>

      {/* <Menu.Item>
          <a href="/RequestPage">문의 게시판</a>
      </Menu.Item> */}
    </Menu>
    )
  }
}

export default LeftMenu