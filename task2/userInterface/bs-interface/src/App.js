import logo from './logo.svg';
import './App.css';
import { Grid, Paper, List, ListItem, ListItemText, Typography, TextField, Button, Container, Divider } from '@mui/material';
import axios from 'axios';
import React, { useState } from 'react';
import TextToText from './textTotext';
import TextToImage from './textToImage';
import { makeStyles } from '@mui/styles';

function App() {

  const [activeComponent, setActiveComponent] = useState('TextToText');

  const handleListItemClick = (component) => {
    setActiveComponent(component);
  };

  return (
    <div className="App">
      <Grid>
        <Grid container direction="row" xs={12}>
          <Grid item xs={2} style={{marginTop:"5%"}}>
            <Paper elevation={0}>
              <List>
                <ListItem selected={activeComponent === 'TextToText'} onClick={() => handleListItemClick('TextToText')}>
                  <Button variant='outlined'>
                    <Typography>
                    Conversational AI
                  </Typography>
                  </Button>
                </ListItem>
                <ListItem selected={activeComponent === 'TextToImage'} onClick={() => handleListItemClick('TextToImage')}>
                  <Button variant='outlined'>
                    <Typography>
                    Text to Image
                  </Typography>
                  </Button>
                </ListItem>
              </List>
            </Paper>
          </Grid>
          <Grid item xs={10}>
            {activeComponent === 'TextToText' && <TextToText />}
            {activeComponent === 'TextToImage' && <TextToImage />}
          </Grid>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;
