import React from 'react'
import { Button, Collapse } from 'antd';
import { useDispatch } from 'react-redux';
import { addToCart } from '../../../../_actions/user_actions';
import { useSelector } from 'react-redux';
import Star from './Star';

const { Panel } = Collapse;

function callback(key) {
    console.log(key);
  }

function ProductInfo(props) {
    let place = ""
    const dispatch = useDispatch();
    const user = useSelector(state => state.user)


    const clickHandler = () => {
        //필요한 정보를 Cart 필드에다가 넣어 준다.
        dispatch(addToCart(props.detail._id))

    }

    if(props.detail.spots === 1){
        place = "학교 앞"
    } else if(props.detail.spots === 2){
        place = "가로수길"
    }
    if (user.userData && !user.userData.isAuth){
        return (
            <div>
                
                <Collapse defaultActiveKey={['1']} onChange={callback}>
                    <Panel header="설명" key="1">
                        <p>{props.detail.description}</p>
                    </Panel>
                    <Panel header="전화번호" key="2">
                        <p>{props.detail.callNumber}</p>
                    </Panel>
                    <Panel header="뷰로의 링크" key="3">
                        <p>{props.detail.reviewlink}</p>
                    </Panel>
                    <Panel header="위치" key="4">
                        <p>{place}</p>
                    </Panel>
                </Collapse>
    
                <br />
                <br />
                <br />
                
    
    
            </div>
        )
    }
    else {
    return (
        <div>
            
            <Collapse defaultActiveKey={['1']} onChange={callback}>
                <Panel header="설명" key="1">
                    <p>{props.detail.description}</p>
                </Panel>
                <Panel header="전화번호" key="2">
                    <p>{props.detail.callNumber}</p>
                </Panel>
                <Panel header="뷰로의 링크" key="3">
                    <p>{props.detail.reviewlink}</p>
                </Panel>
                <Panel header="위치" key="4">
                    <p>{place}</p>
                </Panel>
            </Collapse>

            <br />
            <br />
            <br />
            <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <Button size="large" shape="round" type="danger" onClick={clickHandler}>
                        찜!!
                    </Button>
                </div>

            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <Star
                    userId={localStorage.getItem("userId")}
                    productId={props.productId}
                />
            </div>

        </div>
    )
    }
}

export default ProductInfo
