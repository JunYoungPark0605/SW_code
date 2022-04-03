import React from 'react'
import { Button, Collapse } from 'antd';

const { Panel } = Collapse;

function callback(key) {
    console.log(key);
  }

function WaitInfo(props) {

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
                    <p>{props.detail.spots}</p>
                </Panel>
            </Collapse>

            <br />
            <br />
            <br />
            <div style={{ display: 'flex', justifyContent: 'center' }}>
            </div>


        </div>
    )
}

export default WaitInfo
